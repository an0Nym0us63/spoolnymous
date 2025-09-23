# camera.py
import time
import random
import subprocess
import logging
from threading import Lock
from flask import Response,send_file
from pathlib import Path
import re
import os
import socket
import ssl
import struct

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

# État global (thread-safe) du snapshot
_SNAP_LOCK = Lock()
_SNAP = {
    "ts": 0.0,        # horodatage monotonic de la DERNIÈRE tentative (succès/échec)
    "data": None,     # bytes JPEG du dernier succès
    "ok": False,      # True si 'data' est valide et fraîche
    "fail_count": 0,  # nb d'échecs consécutifs
    "retry_at": 0.0,  # monotonic avant lequel on NE RETENTE PAS
    "last_err": "",   # dernier message d'erreur
}

# Paramètres de cache/backoff (reprennent ceux de app.py)
_SNAP_TTL_OK     = 0.8   # s : TTL en cas de succès (front ~1 Hz → ~1 capture/s)
_FAIL_BASE       = 10.0  # s : premier palier de backoff en cas d'échec
_FAIL_MAX        = 120.0 # s : plafond de backoff
_FAIL_JITTER     = 0.20  # ±20% de jitter
_FFMPEG_TIMEOUTS = 12.0   # délai pour ffmpeg

def _get_printer_model_name() -> str:
    """Récupère le nom modèle via mqtt_bambulab.getPrinterModel()."""
    try:
        from mqtt_bambulab import getPrinterModel  # type: ignore
        info = getPrinterModel() or {}
        return str(info.get("model") or "")
    except Exception:
        return ""

def _get_ip_and_code():
    """Récupère IP et code d'accès via app.get_app_setting, fallback env."""
    try:
        from app import get_app_setting  # type: ignore
    except Exception:
        def get_app_setting(key, default=""):
            return os.environ.get(key, default)
    ip   = get_app_setting("PRINTER_IP", "")
    code = get_app_setting("PRINTER_ACCESS_CODE", "")
    return ip, code

def _snapshot_once_tls6000(timeout_s: float = 5.0) -> bytes:
    """
    Prend UNE image via le flux local TLS port 6000 (protocole Bambu 'bblp').
    Robuste (lecture header 16o + payload) et tolérant sur le JPEG (SOI/EOI).
    """
    ip, access_code = _get_ip_and_code()
    if not ip or not access_code:
        raise RuntimeError("IP ou code d'accès manquant pour TLS6000")

    username = "bblp"
    port = 6000

    # Construire le buffer d'auth tel que fait HA
    auth_data = bytearray()
    auth_data += struct.pack("<I", 0x40)    # '@'\0\0\0
    auth_data += struct.pack("<I", 0x3000)  # \0'0'\0\0
    auth_data += struct.pack("<I", 0)
    auth_data += struct.pack("<I", 0)
    # username (32 octets, ASCII + padding)
    for i in range(len(username)):
        auth_data += struct.pack("<c", username[i].encode("ascii"))
    for _ in range(32 - len(username)):
        auth_data += struct.pack("<x")
    # access_code (32 octets)
    for i in range(len(access_code)):
        auth_data += struct.pack("<c", access_code[i].encode("ascii"))
    for _ in range(32 - len(access_code)):
        auth_data += struct.pack("<x")

    # Contexte TLS permissif (LAN)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Lecture par state machine (header 16o puis payload)
    sock = socket.create_connection((ip, port), timeout=timeout_s)
    try:
        with ctx.wrap_socket(sock, server_hostname=ip) as ssock:
            ssock.settimeout(timeout_s)
            ssock.sendall(auth_data)

            buf = bytearray()
            need_header = True
            payload_size = None

            # petite boucle avec timeout global (réévalué par socket)
            start = time.monotonic()
            while True:
                if time.monotonic() - start > timeout_s:
                    raise TimeoutError("TLS6000 snapshot timeout")

                try:
                    chunk = ssock.recv(4096)
                except ssl.SSLWantReadError:
                    time.sleep(0.05)
                    continue

                if not chunk:
                    # close() côté imprimante ou refus
                    raise RuntimeError("Flux TLS6000 indisponible")

                buf += chunk

                while True:
                    if need_header:
                        if len(buf) < 16:
                            break
                        # 4 octets LE pour la taille payload (corrigé vs [0:3])
                        payload_size = int.from_bytes(buf[0:4], "little")
                        # Optionnel : valider buf[8:12] == b"\x01\x00\x00\x00"
                        buf = buf[16:]
                        need_header = False
                    else:
                        if payload_size is None or len(buf) < payload_size:
                            break
                        img = bytes(buf[:payload_size])
                        buf = buf[payload_size:]
                        # Validation JPEG souple : SOI FFD8, EOI FFD9
                        if not (len(img) >= 2 and img[0] == 0xFF and img[1] == 0xD8):
                            raise RuntimeError("JPEG SOI manquante")
                        if not (len(img) >= 2 and img[-2] == 0xFF and img[-1] == 0xD9):
                            raise RuntimeError("JPEG EOI manquante")
                        return img  # 1 seule image
    finally:
        try:
            sock.close()
        except Exception:
            pass

def _select_providers_for_model(model: str, urls: list[str]):
    """
    Retourne une liste de callables 'providers' à essayer dans l'ordre pour le modèle donné.
    Chaque provider() doit renvoyer des bytes JPEG ou lever une exception.
    """
    model_up = (model or "").upper()

    def _try_rtsp():
        last = None
        for u in urls:
            try:
                return _snapshot_once(u, timeout_s=_FFMPEG_TIMEOUTS)
            except Exception as e:
                last = e
                continue
        raise last or RuntimeError("RTSP providers failed")

    def _try_tls():
        return _snapshot_once_tls6000(timeout_s=5.0)

    if "P1P" in model_up:
        # Pas de caméra de chambre sur P1P
        return []

    # Alignement avec serve_snapshot :
    if any(x in model_up for x in ["H2D", "X1", "X1 CARBON", "X1E"]):
        return [_try_rtsp]
    elif any(x in model_up for x in ["P1S", "A1", "A1 MINI"]):
        return [_try_tls]
    else:
        # Modèle non catégorisé → tenter RTSP puis TLS
        return [_try_rtsp, _try_tls]

def get_camera_urls():
    """
    Construit la/les URL(s) de la caméra à partir de la config appli.
    Retourne (urls: list[str], error: str). Si error != "", config invalide.
    """
    try:
        # import paresseux pour éviter les cycles
        from app import get_app_setting  # type: ignore
    except Exception:
        # fallback éventuel via env, pour ne pas casser en dev
        def get_app_setting(key, default=""):
            return os.environ.get(key, default)

    ip   = get_app_setting("PRINTER_IP", "")
    code = get_app_setting("PRINTER_ACCESS_CODE", "")

    if not ip or not code:
        return [], "IP et/ou code d'accès manquants."

    urls = [
        f"rtsps://bblp:{code}@{ip}:322/streaming/live/1",
        # Tu peux en rajouter ici si besoin, p.ex. un RTSP alternatif en fallback
        # f"rtsp://bblp:{code}@{ip}:322/streaming/live/1",
    ]
    return urls, ""

def svg_fallback(message: str) -> Response:
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 450'>
  <defs><linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>
    <stop offset='0%' stop-color='#f8f9fa'/><stop offset='100%' stop-color='#e9ecef'/>
  </linearGradient></defs>
  <rect width='800' height='450' fill='url(#g)'/>
  <g transform='translate(400,200)' font-family='system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif' text-anchor='middle'>
    <text y='0' font-size='22' fill='#6c757d'>Aperçu caméra indisponible</text>
    <text y='40' font-size='16' fill='#adb5bd'>{message}</text>
  </g>
</svg>"""
    r = Response(svg.encode("utf-8"), mimetype="image/svg+xml")
    r.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
    r.headers["X-Camera-Status"] = "fallback"
    return r

def _snapshot_once(url: str, timeout_s: float = _FFMPEG_TIMEOUTS) -> bytes:
    cmd = [
        "ffmpeg",
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", url,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    out = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout_s, check=True
    )
    if not out.stdout:
        raise RuntimeError("ffmpeg returned no data")
    return out.stdout

def _snapshot_once_auto(urls: list[str]) -> bytes:
    """
    Capture une image en choisissant automatiquement le provider (RTSP/TLS6000)
    selon le modèle d’imprimante. Lève en cas d’échec.
    """
    model = _get_printer_model_name()
    providers = _select_providers_for_model(model, urls)

    if not providers:
        # Cas P1P (ou autre sans caméra)
        raise RuntimeError("Model without chamber camera")

    last_exc = None
    for provider in providers:
        try:
            return provider()
        except subprocess.TimeoutExpired as e:
            last_exc = e
            logger.warning("snapshot auto timeout: %s", e)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr.decode("utf-8", "ignore") if getattr(e, "stderr", None) else "").strip()
            last_exc = RuntimeError(stderr or "ffmpeg error")
            logger.warning("snapshot auto ffmpeg error: %s", stderr or e)
        except Exception as e:
            last_exc = e
            logger.warning("snapshot auto provider error: %s", e)

    raise RuntimeError(f"Snapshot failed on all providers: {last_exc}")

def serve_snapshot() -> Response:
    """
    Tente de capturer une frame sur la première URL disponible.
    - TTL succès : ressert l'image récente sans relancer ffmpeg.
    - Backoff échec : évite les tentatives trop fréquentes.
    - Lock global : garantit au plus 1 ffmpeg à la fois (multi-threads).
    """
    from mqtt_bambulab import isMqttClientConnected
     # 0) Si l'imprimante est hors ligne (MQTT déconnecté), servir un fallback immédiatement.
    try:
        if not isMqttClientConnected():
            offline_path = Path(__file__).resolve().parent / "static" / "offline.png"
            if offline_path.exists():
                r = send_file(str(offline_path), mimetype="image/png")
                r.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
                r.headers["X-Camera-Status"] = "offline"
                return r
            else:
                # fallback ultime si l’image n’existe pas
                r = svg_fallback("Imprimante hors ligne")
                r.headers["X-Camera-Status"] = "offline"
                return r
    except Exception as e:
        logger.warning("Impossible de vérifier l'état MQTT: %s", e)
    urls, err = get_camera_urls()
    if err:
        return svg_fallback(err)
    if not urls:
        return svg_fallback("Aucune URL caméra disponible.")
    now = time.monotonic()

    with _SNAP_LOCK:
        # 1) Cache succès encore frais → renvoyer l'image immédiatement.
        if _SNAP["ok"] and _SNAP["data"] is not None and (now - _SNAP["ts"]) < _SNAP_TTL_OK:
            r = Response(_SNAP["data"], mimetype="image/jpeg")
            r.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
            r.headers["X-Camera-Status"] = "ok"
            r.headers["X-Snapshot-Age"] = f"{now - _SNAP['ts']:.3f}"
            return r

        # 2) En cas d'échec récent, respecter la fenêtre de backoff
        if not _SNAP["ok"] and now < _SNAP["retry_at"] and _SNAP["data"] is not None:
            # on ressert la dernière image (même périmée) + status=stale
            r = Response(_SNAP["data"], mimetype="image/jpeg")
            r.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
            r.headers["X-Camera-Status"] = "stale"
            r.headers["X-Retry-In"] = f"{_SNAP['retry_at'] - now:.3f}"
            return r

        # 3) On va (re)essayer : noter l'instant
        _SNAP["ts"] = now

    # 4) Essayer les URLs, hors lock pour ne pas bloquer le process entier
    model = _get_printer_model_name().upper()
    last_exc = None
    if "P1P" in model:
        last_exc = "Model without chamber camera"
    else:
        def _try_rtsp():
            # tente chaque URL via ffmpeg, renvoie bytes au 1er succès
            last = None
            for u in urls:
                try:
                    return _snapshot_once(u, timeout_s=_FFMPEG_TIMEOUTS)
                except Exception as e:
                    last = e
                    continue
            raise last or RuntimeError("RTSP providers failed")

        def _try_tls():
            return _snapshot_once_tls6000(timeout_s=5.0)

        if any(x in model for x in ["H2D", "X1", "X1 CARBON", "X1E"]):
            providers = [_try_rtsp]
        elif any(x in model for x in ["P1S", "A1", "A1 MINI"]):
            providers = [_try_tls]
        else:
            # Modèle inconnu : tenter RTSP puis TLS
            providers = [_try_rtsp, _try_tls]
    if providers:
        for provider in providers:
            try:
                data = provider()
                # Succès → mettre à jour l'état (sous lock)
                with _SNAP_LOCK:
                    _SNAP["data"] = data
                    _SNAP["ok"] = True
                    _SNAP["fail_count"] = 0
                    _SNAP["retry_at"] = 0.0
                    _SNAP["last_err"] = ""
                    r = Response(data, mimetype="image/jpeg")
                    r.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
                    r.headers["X-Camera-Status"] = "ok"
                    r.headers["X-Snapshot-Age"] = "0.000"
                    r.headers["X-Provider"] = "tls6000" if provider.__name__ == "_try_tls" else "rtsp"
                    return r
            except subprocess.TimeoutExpired as e:
                last_exc = "Timeout"
                logger.warning("snapshot camera timeout: %s", e)
            except subprocess.CalledProcessError as e:
                stderr = (e.stderr.decode("utf-8", "ignore") if getattr(e, "stderr", None) else "").strip()
                last_exc = "Camera ffmpeg error"
                logger.warning("snapshot camera ffmpeg error: %s", stderr or e)
            except Exception as e:
                last_exc = f"Provider {provider.__name__} failed"
                logger.warning("snapshot camera provider error: %s", e)

    # 5) Tous les essais ont échoué → backoff + servir fallback
    with _SNAP_LOCK:
        _SNAP["ok"] = False
        _SNAP["fail_count"] = min(_SNAP["fail_count"] + 1, 999999)

        # Limite de fail_count pour éviter OverflowError lors de 2 ** x
        safe_fail_count = min(_SNAP["fail_count"], 32)
        base = min(_FAIL_BASE * (2 ** (safe_fail_count - 1)), _FAIL_MAX)
        
        jitter = base * _FAIL_JITTER * (2 * random.random() - 1.0)
        wait_s = max(1.0, base + jitter)
        _SNAP["retry_at"] = time.monotonic() + wait_s
        _SNAP["last_err"] = str(last_exc or "error")

    msg = _SNAP["last_err"] or "Erreur snapshot"
    r = svg_fallback(msg if "P1P" not in model else "Caméra chambre indisponible sur ce modèle")
    r.headers["X-Retry-In"] = f"{wait_s:.3f}"
    r.headers["X-Model"] = model or "Unknown"
    return r

def _sanitize_filename(name: str) -> str:
    """
    Nettoie un nom de fichier (sans extension) : remplace tout caractère
    non autorisé par '_', et supprime les points initiaux.
    """
    if not name:
        return "snapshot"
    name = name.strip()
    # retirer extension si l'appelant en a mis une par erreur
    name = name.split(".")[0]
    name = _SAFE_NAME_RE.sub("_", name)
    # éviter les noms vides/cachés
    if not name or name.startswith("."):
        name = "snapshot"
    return name

def snapshot_to_print_file(print_id: str | int, filename_no_ext: str) -> tuple[str, str]:
    """
    Capture un snapshot (JPEG) et l'enregistre dans:
      static/uploads/prints/{print_id}/{filename}.jpg

    Retourne (absolute_path, static_url). Lève en cas d'échec capture.
    """
    urls, err = get_camera_urls()
    if err:
        raise RuntimeError(err)
    if not urls:
        raise RuntimeError("Aucune URL caméra disponible.")
    # 1) capturer (essaie chaque URL)
    data = None
    last_exc = None
    for u in urls:
        try:
            data = _snapshot_once_auto(urls)
            break
        except Exception as e:
            last_exc = e
            continue
    if data is None:
        raise RuntimeError(f"Snapshot failed on all URLs: {last_exc}")

    # 2) chemin
    base_dir = Path(__file__).resolve().parent
    target_dir = base_dir / "static" / "uploads" / "prints" / str(print_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    basename = _sanitize_filename(filename_no_ext)
    target_path = target_dir / f"{basename}.jpg"

    # 3) écrire
    target_path.write_bytes(data)

    # 4) URL statique
    rel_url = f"/static/uploads/prints/{print_id}/{basename}.jpg"
    return str(target_path), rel_url


