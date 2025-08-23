# camera.py
import time
import random
import subprocess
import logging
from threading import Lock
from flask import Response
from pathlib import Path
import re
import os

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
_FFMPEG_TIMEOUTS = 6.0   # délai pour ffmpeg

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

def serve_snapshot() -> Response:
    """
    Tente de capturer une frame sur la première URL disponible.
    - TTL succès : ressert l'image récente sans relancer ffmpeg.
    - Backoff échec : évite les tentatives trop fréquentes.
    - Lock global : garantit au plus 1 ffmpeg à la fois (multi-threads).
    """
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
    last_exc = None
    for u in urls:
        try:
            data = _snapshot_once(u, timeout_s=_FFMPEG_TIMEOUTS)
            # Succès → mettre à jour l'état (sous lock)
            with _SNAP_LOCK:
                _SNAP["data"] = data
                _SNAP["ok"] = True
                _SNAP["fail_count"] = 0
                _SNAP["retry_at"] = 0.0
                _SNAP["last_err"] = ""
                # réponse fraîche
                r = Response(data, mimetype="image/jpeg")
                r.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
                r.headers["X-Camera-Status"] = "ok"
                r.headers["X-Snapshot-Age"] = "0.000"
                return r

        except subprocess.TimeoutExpired as e:
            last_exc = e
            logger.warning("snapshot camera timeout on %s: %s", u, e)

        except subprocess.CalledProcessError as e:
            stderr = (e.stderr.decode("utf-8", "ignore") if e.stderr else "").strip()
            last_exc = RuntimeError(stderr or str(e))
            logger.warning("snapshot camera ffmpeg error on %s: %s", u, stderr or e)

        except Exception as e:
            last_exc = e
            logger.warning("snapshot camera unexpected error on %s: %s", u, e)

    # 5) Tous les essais ont échoué → backoff + servir fallback
    with _SNAP_LOCK:
        _SNAP["ok"] = False
        _SNAP["fail_count"] = min(_SNAP["fail_count"] + 1, 999999)
        base = min(_FAIL_BASE * (2 ** (_SNAP["fail_count"] - 1)), _FAIL_MAX)
        jitter = base * _FAIL_JITTER * (2 * random.random() - 1.0)  # ±jitter
        wait_s = max(1.0, base + jitter)
        _SNAP["retry_at"] = time.monotonic() + wait_s
        _SNAP["last_err"] = str(last_exc) if last_exc else "unknown error"

    msg = _SNAP["last_err"] or "Erreur snapshot"
    r = svg_fallback(msg)
    r.headers["X-Retry-In"] = f"{wait_s:.3f}"
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
    time.sleep(5)
    # 1) capturer (essaie chaque URL)
    data = None
    last_exc = None
    for u in urls:
        try:
            data = _snapshot_once(u, timeout_s=_FFMPEG_TIMEOUTS)
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
