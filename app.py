import json
import traceback
import logging
from exceptions import ApplicationError
import uuid
import math
from datetime import datetime, timedelta
import time
import os
import re
from collections import defaultdict,deque
from urllib.parse import urlencode
import requests
import secrets
from queue import Queue, Empty, Full
import threading
from threading import Lock
from flask import Response
from itertools import count
import subprocess
import signal
import random

from flask_login import LoginManager, login_required
from auth import auth_bp, User, get_stored_user
from flask import flash,Flask, request, render_template, redirect, url_for,jsonify,g, make_response,send_from_directory, abort,stream_with_context, Response, abort,current_app
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

from config import AUTO_SPEND, EXTERNAL_SPOOL_AMS_ID, EXTERNAL_SPOOL_ID, PRINTER_NAME,get_app_setting,set_app_setting
from filament import generate_filament_brand_code, generate_filament_temperatures
from frontend_utils import color_is_dark
from messages import AMS_FILAMENT_SETTING
from mqtt_bambulab import getLastAMSConfig, publish, getMqttClient, setActiveTray, isMqttClientConnected, init_mqtt, getPrinterModel,insert_manual_print
from spoolman_client import patchExtraTags, getSpoolById, consumeSpool, archive_spool, reajust_spool
from spoolman_service import augmentTrayDataWithSpoolMan, trayUid, getSettings,fetchSpools
from print_history import get_prints_with_filament, update_filament_spool, get_filament_for_slot,get_distinct_values,update_print_filename,get_filament_for_print, delete_print, get_tags_for_print, add_tag_to_print, remove_tag_from_print,update_filament_usage,update_print_history_field,create_print_group,get_print_groups,update_print_group_field,update_group_created_at,get_group_id_of_print,get_statistics,adjustDuration,set_group_primary_print,set_sold_info,recalculate_print_data, recalculate_group_data,cleanup_orphan_data,get_latest_print,get_all_tray_spool_mappings,set_tray_spool_map,delete_all_tray_spool_mappings, get_tags_for_prints, get_tags_for_group, add_tag_to_group, remove_tag_from_group
from globals import PRINTER_STATUS, PRINTER_STATUS_LOCK
from installations import load_installations
from switcher import switch_bp
from objects import get_available_units, create_objects_from_source, list_objects, get_tags_for_objects, rename_object, delete_object,get_object_counts_by_parent,update_object_sale,clear_object_sale,update_object_comment,summarize_objects,add_object_tag, remove_object_tag,list_accessories, get_accessory, create_accessory, add_accessory_stock, link_accessory_to_object, unlink_accessory_from_object, list_object_accessories,remove_accessory_stock, delete_accessory,set_accessory_image_path,list_objects_using_accessory,rename_accessory

logging.basicConfig(
    level=logging.DEBUG,  # ou DEBUG si tu veux plus de détails
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

for name in ("urllib3", "urllib3.connectionpool", "requests.packages.urllib3"):
    lg = logging.getLogger(name)
    lg.setLevel(logging.WARNING)

COLOR_FAMILIES = {
    # Neutres
    'Black': (0, 0, 0),
    'White': (255, 255, 255),
    'Grey': (160, 160, 160),

    # Rouges et dérivés
    'Red': (220, 20, 60),         # Crimson
    'Dark Red': (139, 0, 0),      # sombre
    'Pink': (255, 182, 193),      # pastel
    'Magenta': (255, 0, 255),     # fuchsia
    'Brown': (150, 75, 0),        # chocolat

    # Jaunes et dérivés
    'Yellow': (255, 220, 0),      # chaud
    'Gold': (212, 175, 55),       # doré
    'Orange': (255, 140, 0),      # foncé

    # Verts
    'Green': (80, 200, 120),      # gazon
    'Dark Green': (0, 100, 0),    # forêt
    'Lime': (191, 255, 0),        # fluo
    'Teal': (0, 128, 128),        # turquoise

    # Bleus et violets
    'Blue': (100, 150, 255),      # clair
    'Navy': (0, 0, 128),          # foncé
    'Cyan': (0, 255, 255),        # turquoise clair
    'Lavender': (230, 230, 250),  # violet pastel
    'Purple': (160, 32, 240),
    'Dark Purple': (90, 60, 120), # violet foncé
}

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')[:6]
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    r /= 255
    g /= 255
    b /= 255

    def gamma_correct(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r = gamma_correct(r)
    g = gamma_correct(g)
    b = gamma_correct(b)

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return (l, a, b)


def color_distance(hex1: str, hex2: str) -> float:
    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)
    lab1 = rgb_to_lab(*rgb1)
    lab2 = rgb_to_lab(*rgb2)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def two_closest_families(hex_color: str, threshold: float = 60.0) -> list[str]:
    """
    Retourne la famille la plus proche et la deuxième si sa distance est < threshold.
    """
    distances = {
        family: color_distance(hex_color, '#{:02X}{:02X}{:02X}'.format(*rgb))
        for family, rgb in COLOR_FAMILIES.items()
    }
    sorted_families = sorted(distances.items(), key=lambda x: x[1])

    result = [sorted_families[0][0]]  # toujours la plus proche
    if sorted_families[1][1] <= threshold:
        result.append(sorted_families[1][0])
    return result

def compute_pagination_pages(page, total_pages, window=2, max_buttons=5):
    pages = []
    if total_pages <= max_buttons:
        pages = list(range(1, total_pages + 1))
    else:
        pages = [1]

        start = max(2, page - window)
        end = min(total_pages - 1, page + window)

        if start > 2:
            pages.append('…')

        pages.extend(range(start, end + 1))

        if end < total_pages - 1:
            pages.append('…')

        pages.append(total_pages)

    return pages

DEFAULT_KEEP_KEYS = {
    'print_history': ["page", "filament_type", "color", "filament_id", "status", "search"],
    'filaments': ["page", "search", "color", "sort", "include_archived",
                  "assign_print_id", "assign_filament_index", "assign_filament_type","assign_filament_id","assign_sold_filter", "assign_color","assign_page", "assign_search","assign_status", "filament_usage", "ams", "tray","is_assign_mode","tray_uuid","tray_info_idx","tray_color"],
    'stats': ["period","search","filament_type","color","origin","origin_label","current_label"],
    'objects_page': ["page", "search", "source_type", "sale_filter", "focus_object_id"],
}

def _merge_context_args(keep=None, drop=None, endpoint=None, **new_args):
    """
    Fusionne les arguments GET et certains POST explicitement autorisés
    avec des nouveaux paramètres, en nettoyant les clés vides.
    Les arguments GET ont priorité sur les POST en cas de doublon.
    """

    def is_meaningful(val):
        if isinstance(val, list):
            val = list(dict.fromkeys(v for v in val if v not in [None, ""]))
            return val if val else None
        return val if val not in [None, ""] else None

    current_args = {}
    route = endpoint or request.endpoint
    effective_keep = set(DEFAULT_KEEP_KEYS.get(route, []))
    if keep is not None:
        effective_keep.update(keep)

    for k in request.args:
        if k in effective_keep:
            values = request.args.getlist(k)
            cleaned = is_meaningful(values if len(values) > 1 else values[0])
            if cleaned is not None:
                current_args[k] = cleaned

    if request.method == 'POST':
        for k in request.form:
            if k in effective_keep:
                values = request.form.getlist(k)
                cleaned = is_meaningful(values if len(values) > 1 else values[0])
                if cleaned is not None:
                    current_args[k] = cleaned

    # new_args peuvent écraser, donc on applique le même nettoyage
    cleaned_new_args = {
        k: is_meaningful(v) for k, v in new_args.items()
        if is_meaningful(v) is not None
    }

    merged = {**current_args, **cleaned_new_args}
    return merged
    
def redirect_with_context(endpoint, keep=None, drop=None, **new_args):
    """
    Redirige vers une route en conservant certains arguments (GET et POST filtrés).

    Args:
        endpoint (str): nom de la route Flask.
        keep (list[str], optional): clés supplémentaires à conserver en plus de DEFAULT_KEEP_KEYS.
        drop (list[str], optional): liste noire des clés à exclure. Ignoré si keep est fourni.
        **new_args: clés/valeurs à injecter ou à écraser explicitement.

    Returns:
        werkzeug.wrappers.Response: redirection HTTP propre avec les bons paramètres.
    """
    combined_args = _merge_context_args(keep=keep, drop=drop, endpoint=endpoint, **new_args)
    query = urlencode(combined_args, doseq=True)
    return redirect(url_for(endpoint) + ('?' + query if query else ''))

def filtered_args_for_template(keep=None, drop=None, **overrides):
    """
    Génère un dictionnaire de paramètres filtrés (comme pour redirection)
    à utiliser dans render_template(..., args=...)
    """
    return _merge_context_args(keep=keep, drop=drop, **overrides)

def parse_print_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except:
        return datetime.min  # en cas de date invalide, met tout en bas



LOG_BUFFER_MAX = 1000                                # lignes conservées
_log_buffer = deque(maxlen=LOG_BUFFER_MAX)           # (idx, levelno, message)
_log_lock = Lock()
_log_counter = count(1)                              # index auto-incrémenté des lignes

LEVELS_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

class InMemoryLogHandler(logging.Handler):
    """Handler simple : pousse chaque log formaté dans un buffer en mémoire."""
    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        idx = next(_log_counter)
        with _log_lock:
            _log_buffer.append((idx, record.levelno, msg))

# Installe le handler si absent
_root = logging.getLogger()

if not any(isinstance(h, InMemoryLogHandler) for h in _root.handlers):
    _root.addHandler(InMemoryLogHandler())
  
_SNAP_LOCK = Lock()
_SNAP = {
    "ts": 0.0,        # horodatage monotonic de la DERNIÈRE tentative (succès/échec)
    "data": None,     # bytes JPEG du dernier succès
    "ok": False,      # True si 'data' est valide et fraîche
    "fail_count": 0,  # nb d'échecs consécutifs
    "retry_at": 0.0,  # monotonic avant lequel on NE RETENTE PAS
    "last_err": "",   # dernier message d'erreur
}

_SNAP_TTL_OK     = 0.8   # secondes : TTL en cas de succès (front ~1 Hz → ~1 capture/s)
_FAIL_BASE       = 10.0  # secondes : premier palier de backoff en cas d'échec
_FAIL_MAX        = 120.0 # plafond de backoff
_FAIL_JITTER     = 0.20  # ±20% de jitter
_FFMPEG_TIMEOUTS = 6.0   # cohérent avec _snapshot_once(...)

def _svg_fallback(message: str) -> Response:
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 450'>
  <defs><linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>
    <stop offset='0%' stop-color='#222'/><stop offset='100%' stop-color='#111'/>
  </linearGradient></defs>
  <rect width='100%' height='100%' fill='url(#g)'/>
  <g fill='none' stroke='#444' stroke-width='4'>
    <rect x='150' y='120' width='500' height='300' rx='16' ry='16'/>
    <circle cx='400' cy='270' r='60' stroke='#666'/><circle cx='400' cy='270' r='20' stroke='#888'/>
  </g>
  <g fill='#bbb' font-family='ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto' text-anchor='middle'>
    <text x='400' y='80' font-size='28'>Caméra indisponible</text>
    <text x='400' y='120' font-size='16' fill='#999'>{message}</text>
  </g>
</svg>"""
    r = Response(svg.encode("utf-8"), mimetype="image/svg+xml")
    r.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
    r.headers["X-Camera-Status"] = "error"
    return r

def _next_backoff(fails: int) -> float:
    """Calcule la pause avant le prochain essai après 'fails' échecs consécutifs."""
    if fails <= 0:
        base = _FAIL_BASE
    else:
        base = min(_FAIL_BASE * (2 ** (fails - 1)), _FAIL_MAX)
    jitter = base * random.uniform(-_FAIL_JITTER, _FAIL_JITTER)
    return max(1.0, base + jitter)

def _serve_snapshot(urls) -> Response:
    """
    Tente de capturer une frame sur la première URL disponible.
    - TTL succès : ressert l'image récente sans relancer ffmpeg.
    - TTL échec  : impose un backoff progressif avant toute nouvelle tentative.
    - Lock global: garantit au plus 1 ffmpeg à la fois (multi-threads).
    """
    now = time.monotonic()

    with _SNAP_LOCK:
        # 1) Cache succès encore frais → renvoyer l'image immédiatement.
        if _SNAP["ok"] and _SNAP["data"] is not None and (now - _SNAP["ts"]) < _SNAP_TTL_OK:
            r = Response(_SNAP["data"], mimetype="image/jpeg")
            r.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
            r.headers["X-Camera-Status"] = "ok"
            r.headers["X-Snapshot-Age"] = f"{now - _SNAP['ts']:.3f}"
            return r

        # 2) En période de backoff après échec → ne pas retenter.
        if (not _SNAP["ok"]) and now < _SNAP["retry_at"]:
            remaining = _SNAP["retry_at"] - now
            r = _svg_fallback(_SNAP["last_err"] or "Caméra indisponible — prochain essai différé")
            r.headers["Retry-After"] = f"{int(remaining)}"
            r.headers["X-Retry-After-Seconds"] = f"{remaining:.3f}"
            r.headers["X-Camera-Status"] = "error"
            return r

        # 3) Autorisé à retenter maintenant → sérialiser ffmpeg (lock maintenu).
        last_exc = None
        for u in urls:
            try:
                jpg = _snapshot_once(u, timeout_s=_FFMPEG_TIMEOUTS)
                # Succès: mémoriser état frais (timestamp post-capture).
                now2 = time.monotonic()
                _SNAP.update(ts=now2, data=jpg, ok=True, fail_count=0, retry_at=0.0, last_err="")
                r = Response(jpg, mimetype="image/jpeg")
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
                logger.warning("snapshot camera fail on %s: %s", u, e)

        # 4) Tous les essais ont échoué → activer backoff et renvoyer fallback.
        now2 = time.monotonic()
        _SNAP["ok"] = False
        _SNAP["ts"] = now2
        _SNAP["data"] = None
        _SNAP["fail_count"] = _SNAP.get("fail_count", 0) + 1
        _SNAP["last_err"] = (str(last_exc) if last_exc else "Erreur inconnue")[:200]
        delay = _next_backoff(_SNAP["fail_count"])
        _SNAP["retry_at"] = now2 + delay

        r = _svg_fallback(f"{_SNAP['last_err']} — nouvel essai dans ~{int(delay)}s")
        r.headers["Retry-After"] = f"{int(delay)}"
        r.headers["X-Camera-Status"] = "error"
        return r

            
def _snapshot_once(url: str, timeout_s: float = 6.0) -> bytes:
    cmd = [
        "ffmpeg",
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", url,
        "-frames:v", "1",
        "-f", "image2pipe",                       # <- comme ton test, mais vers stdout
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    out = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout_s, check=True
    )
    if not out.stdout:
        raise RuntimeError("ffmpeg: sortie vide")
    return out.stdout

init_mqtt()

app = Flask(__name__)
logger = logging.getLogger(__name__)
secret_key = get_app_setting("SECRET_KEY", "")
if not secret_key:
    secret_key = secrets.token_hex(32)
    set_app_setting("SECRET_KEY", secret_key)
app.secret_key = secret_key

login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # redirige vers /login si non connecté
login_manager.init_app(app)
app.config["PREFERRED_URL_SCHEME"] = "https"
app.register_blueprint(switch_bp)
app.config.update(
    PREFERRED_URL_SCHEME='https',          # url_for(..., _external=True) → https
    SESSION_COOKIE_SAMESITE='None',        # cookies utilisables en iframe (tiers)
    SESSION_COOKIE_SECURE=True             # requis avec SameSite=None
)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

ACCESSORY_UPLOAD_DIR = os.path.join(app.static_folder, "uploads", "accessories")
os.makedirs(ACCESSORY_UPLOAD_DIR, exist_ok=True)

ALLOWED_IMG_EXT = {"png", "jpg", "jpeg", "webp", "gif"}
def _allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMG_EXT

def _abs_static_path(rel_path: str) -> str:
    # rel_path ex: 'uploads/accessories/acc_12_xxx.png'
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    return os.path.join(app.static_folder, rel_path)

@app.get("/camera/snapshot")
def camera_snapshot():
    ip   = get_app_setting("PRINTER_IP", "")
    code = get_app_setting("PRINTER_ACCESS_CODE", "")
    if not ip or not code:
        return _svg_fallback("IP et/ou code d'accès manquants.")

    # Essaye /1 puis /0 (certains firmwares n'exposent qu'un seul track)
    urls= [
        f"rtsps://bblp:{code}@{ip}:322/streaming/live/1"
    ]
    return _serve_snapshot(urls)

@app.route("/logs")
@login_required
def logs_page():
    # Page HTML (UI)
    return render_template("logs.html", page_title="Logs")

@app.route("/logs/poll")
@login_required
def logs_poll():
    """Retourne les nouvelles lignes depuis ?since=..., filtrées par ?level et ?q (regex)."""
    level_name = (request.args.get("level") or "INFO").upper()
    min_level  = LEVELS_MAP.get(level_name, logging.INFO)
    q          = request.args.get("q") or ""
    rx         = re.compile(q) if q else None
    since      = int(request.args.get("since") or 0)

    with _log_lock:
        lines = [(idx, msg) for (idx, lvl, msg) in _log_buffer
                 if idx > since and lvl >= min_level and (rx is None or rx.search(msg))]

    # Borne anti-burst
    MAX_LINES = 500
    if len(lines) > MAX_LINES:
        lines = lines[-MAX_LINES:]

    last = lines[-1][0] if lines else since
    return jsonify({"lines": [m for _, m in lines], "last": last})
# --- [LOGS BUFFER/POLL] Fin ---

@app.before_request
def strip_theme_param():
    # On ne touche qu'aux GET avec un endpoint FLASK
    if request.method == 'GET' and request.endpoint and 'theme' in request.args and request.endpoint != 'static':
        args = request.args.to_dict(flat=True)
        args.pop('theme', None)
        return redirect(url_for(request.endpoint, **(request.view_args or {}), **args), code=302)
        
@app.url_value_preprocessor
def _pull_origin(endpoint, values):
    g._origin = request.args.get('origin')
    g._origin_label = request.args.get('origin_label')
    g._current_label = request.args.get('current_label')  # 👈

@app.url_defaults
def _push_origin(endpoint, values):
    if endpoint == 'static':
        return
    if g.get('_origin') and 'origin' not in values:
        values['origin'] = g._origin
    if g.get('_origin_label') and 'origin_label' not in values:
        values['origin_label'] = g._origin_label
    if g.get('_current_label') and 'current_label' not in values:  # 👈
        values['current_label'] = g._current_label

@login_manager.user_loader
def load_user(user_id: str):
    # Invité auto-connecté via /guest/<token>
    if user_id.startswith("guest:"):
        return User(user_id, role="guest")

    # Utilisateur stocké (admin configuré)
    data = get_stored_user()
    if data and user_id in data:
        return User(user_id, role="user")

    # Admin par défaut (fallback)
    if user_id == app.config.get("DEFAULT_ADMIN_USERNAME", "admin"):
        return User(user_id, role="user")

    return None

@app.before_request
def detect_webview():
    g.is_webview = request.cookies.get('webview') == '1'

@app.before_request
def require_login():
    from flask_login import current_user
    exempt_routes = {
        'auth.login',
        'auth.autologin_token',   # tu l'avais déjà
        'auth.guest_autologin',   # <-- AJOUT pour le lien invité /guest/<token>
        'static',
    }
    # ⚠️ Retire 'auth.settings' de la whitelist pour protéger la page
    if request.endpoint not in exempt_routes and not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    
@app.template_filter('datetimeformat')
def datetimeformat(value, locale='fr'):
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%d/%m/%Y %H:%M")
    
@app.template_filter('hm_format')
def hm_format(hours: float):
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h}h {m:02d}min"

@app.context_processor
def inject_installations():
    return {
        "installations": load_installations()
    }

@app.context_processor
def frontend_utilities():
    def url_with_args(**kwargs):
        args = _merge_context_args(**kwargs)
        return url_for(request.endpoint) + ('?' + urlencode(args, doseq=True) if args else '')

    return dict(
        SPOOLMAN_BASE_URL=get_app_setting("SPOOLMAN_BASE_URL",""),
        AUTO_SPEND=AUTO_SPEND,
        color_is_dark=color_is_dark,
        BASE_URL=get_app_setting("BASE_URL",""),
        EXTERNAL_SPOOL_AMS_ID=EXTERNAL_SPOOL_AMS_ID,
        EXTERNAL_SPOOL_ID=EXTERNAL_SPOOL_ID,
        PRINTER_MODEL=getPrinterModel(),
        PRINTER_NAME=PRINTER_NAME,
        url_with_args=url_with_args
    )

@app.errorhandler(ApplicationError)
def handle_application_error(error):
    logger.error("ApplicationError capturée :\n%s", traceback.format_exc())
    return render_template("error.html", exception=str(error)), 500
    
@app.route("/issue")
def issue():
  if not isMqttClientConnected():
    return render_template('error.html', exception="L'imprimante est elle allumée ? Avez vous renseigné les paramètres ?")
    
  ams_id = request.args.get("ams")
  tray_id = request.args.get("tray")
  if not all([ams_id, tray_id]):
    return render_template('error.html', exception="Missing AMS ID, or Tray ID.")

  fix_ams = None

  spool_list = fetchSpools()
  last_ams_config = getLastAMSConfig()
  if ams_id == EXTERNAL_SPOOL_AMS_ID:
    fix_ams = last_ams_config.get("vt_tray", {})
  else:
    for ams in last_ams_config.get("ams", []):
      if ams["id"] == ams_id:
        fix_ams = ams
        break

  active_spool = None
  for spool in spool_list:
    if spool.get("extra") and spool["extra"].get("active_tray") and spool["extra"]["active_tray"] == json.dumps(trayUid(ams_id, tray_id)):
      active_spool = spool
      break

  #TODO: Determine issue
  #New bambulab spool
  #Tray empty, but spoolman has record
  #Extra tag mismatch?
  #COLor/type mismatch

  return render_template('issue.html', fix_ams=fix_ams, active_spool=active_spool)

@app.route("/spool_info")
def spool_info():
  if not isMqttClientConnected():
    return render_template('error.html', exception="L'imprimante est elle allumée ? Avez vous renseigné les paramètres ?")
    
  try:
    tag_id = request.args.get("tag_id", "-1")
    spool_id = request.args.get("spool_id", -1)
    last_ams_config = getLastAMSConfig()
    ams_data = last_ams_config.get("ams", [])
    vt_tray_data = last_ams_config.get("vt_tray", {})
    spool_list = fetchSpools()
    
    issue = False
    #TODO: Fix issue when external spool info is reset via bambulab interface
    augmentTrayDataWithSpoolMan(spool_list, vt_tray_data, trayUid(EXTERNAL_SPOOL_AMS_ID, EXTERNAL_SPOOL_ID))
    issue |= vt_tray_data["issue"]

    for ams in ams_data:
      for tray in ams["tray"]:
        augmentTrayDataWithSpoolMan(spool_list, tray, trayUid(ams["id"], tray["id"]))
        issue |= tray["issue"]

    if not tag_id:
      return render_template('error.html', exception="TAG ID is required as a query parameter (e.g., ?tag_id=RFID123)")

    spools = fetchSpools()
    current_spool = None
    for spool in spools:
      if spool['id'] == int(spool_id):
        current_spool = spool
        break

      if not spool.get("extra", {}).get("tag"):
        continue

      tag = json.loads(spool["extra"]["tag"])
      if tag != tag_id:
        continue

      current_spool = spool

    if current_spool:
      # TODO: missing current_spool
      return render_template('spool_info.html', tag_id=tag_id, current_spool=current_spool, ams_data=ams_data, vt_tray_data=vt_tray_data)
    else:
      return render_template('error.html', exception="Spool not found")
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))


@app.route("/tray_load")
def tray_load():
  if not isMqttClientConnected():
    return render_template('error.html', exception="L'imprimante est elle allumée ? Avez vous renseigné les paramètres ?")
  
  tag_id = request.args.get("tag_id")
  ams_id = request.args.get("ams")
  tray_id = request.args.get("tray")
  spool_id = request.args.get("spool_id")

  if not all([ams_id, tray_id, spool_id]):
    return render_template('error.html', exception="Missing AMS ID, or Tray ID or spool_id.")

  try:
    # Update Spoolman with the selected tray
    spool_data = getSpoolById(spool_id)
    setActiveTray(spool_id, spool_data["extra"], ams_id, tray_id)
    setActiveSpool(ams_id, tray_id, spool_data)

    return redirect(url_for('home', success_message=f"Updated Spool ID {spool_id} with TAG id {tag_id} to AMS {ams_id}, Tray {tray_id}."))
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))

def setActiveSpool(ams_id, tray_id, spool_data):
  if not isMqttClientConnected():
    return render_template('error.html', exception="L'imprimante est elle allumée ? Avez vous renseigné les paramètres ?")
  
  ams_message = AMS_FILAMENT_SETTING
  ams_message["print"]["sequence_id"] = 0
  ams_message["print"]["ams_id"] = int(ams_id)
  ams_message["print"]["tray_id"] = int(tray_id)
  
  if "color_hex" in spool_data["filament"]:
    ams_message["print"]["tray_color"] = spool_data["filament"]["color_hex"].upper() + "FF"
  else:
    ams_message["print"]["tray_color"] = spool_data["filament"]["multi_color_hexes"].split(',')[0].upper() + "FF"
      
  if "nozzle_temperature" in spool_data["filament"]["extra"]:
    nozzle_temperature_range = spool_data["filament"]["extra"]["nozzle_temperature"].strip("[]").split(",")
    ams_message["print"]["nozzle_temp_min"] = int(nozzle_temperature_range[0])
    ams_message["print"]["nozzle_temp_max"] = int(nozzle_temperature_range[1])
  else:
    nozzle_temperature_range_obj = generate_filament_temperatures(spool_data["filament"]["material"],
                                                                  spool_data["filament"]["vendor"]["name"])
    ams_message["print"]["nozzle_temp_min"] = int(nozzle_temperature_range_obj["filament_min_temp"])
    ams_message["print"]["nozzle_temp_max"] = int(nozzle_temperature_range_obj["filament_max_temp"])

  ams_message["print"]["tray_type"] = spool_data["filament"]["material"]

  filament_brand_code = {}
  filament_brand_code["brand_code"] = spool_data["filament"]["extra"].get("filament_id", "").strip('"')
  filament_brand_code["sub_brand_code"] = ""

  if filament_brand_code["brand_code"] == "":
    filament_brand_code = generate_filament_brand_code(spool_data["filament"]["material"],
                                                      spool_data["filament"]["vendor"]["name"],
                                                      spool_data["filament"]["extra"].get("type", ""))
    
  ams_message["print"]["tray_info_idx"] = filament_brand_code["brand_code"]

  # TODO: test sub_brand_code
  # ams_message["print"]["tray_sub_brands"] = filament_brand_code["sub_brand_code"]
  ams_message["print"]["tray_sub_brands"] = ""

  print(ams_message)
  #publish(getMqttClient(), ams_message)
@app.route("/")
def home():
  if not isMqttClientConnected():
    return render_template('error.html', exception="L'imprimante est elle allumée ? Avez vous renseigné les paramètres ?")
    
  try:
    with PRINTER_STATUS_LOCK:
        status_copy = dict(PRINTER_STATUS)
    last_ams_config = getLastAMSConfig()
    ams_data = last_ams_config.get("ams", [])
    vt_tray_data = last_ams_config.get("vt_tray", {})
    spool_list = fetchSpools()
    success_message = request.args.get("success_message")
    
    issue = False
    #TODO: Fix issue when external spool info is reset via bambulab interface
    augmentTrayDataWithSpoolMan(spool_list, vt_tray_data, trayUid(EXTERNAL_SPOOL_AMS_ID, EXTERNAL_SPOOL_ID))
    issue |= vt_tray_data["issue"]

    for ams in ams_data:
      for tray in ams["tray"]:
        augmentTrayDataWithSpoolMan(spool_list, tray, trayUid(ams["id"], tray["id"]))
        issue |= tray["issue"]
      location = ''
      LOCATION_MAPPING=get_app_setting('LOCATION_MAPPING','')
      if LOCATION_MAPPING != '' :
        d = dict(item.split(":", 1) for item in LOCATION_MAPPING.split(";"))
        ams_name='AMS_'+str(ams["id"])
        if ams_name in d:
            location = d[ams_name]
      ams['location']=location
    AMS_ORDER=get_app_setting("AMS_ORDER","")
    if AMS_ORDER != '':
      mapping = {int(k): int(v) for k, v in (item.split(":") for item in AMS_ORDER.split(";"))}
      reordered = [None] * len(ams_data)
      for src_index, dst_index in mapping.items():
          reordered[dst_index] = ams_data[src_index]
      ams_data=reordered
    latest = get_latest_print()
    if latest:
        status_copy["printName"] = latest["file_name"]
        if "image_file" in latest:
            status_copy["thumbnail"] = latest["image_file"]
        else:
            status_copy["thumbnail"] = None
    else:
        status_copy["thumbnail"] = None
        status_copy["printName"] = None
    # Nouveau : si ?webview=1 → on met le cookie
    resp = make_response(render_template(
        'index.html',
        success_message=success_message,
        ams_data=ams_data,
        vt_tray_data=vt_tray_data,
        issue=issue,
        page_title="Accueil",
        printer_status=status_copy
    ))

    if request.args.get("webview") == "1":
        resp.set_cookie("webview", "1")

    return resp
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))

def sort_spools(spools):
  def condition(item):
    # Ensure the item has an "extra" key and is a dictionary
    if not isinstance(item, dict) or "extra" not in item or not isinstance(item["extra"], dict):
      return False

    # Check the specified condition
    return item["extra"].get("tag") or item["extra"].get("tag") == ""

  # Sort with the custom condition: False values come first
  return sorted(spools, key=lambda spool: bool(condition(spool)))

@app.route("/assign_tag")
def assign_tag():
  if not isMqttClientConnected():
    return render_template('error.html', exception="L'imprimante est elle allumée ? Avez vous renseigné les paramètres ?")
    
  try:
    spools = sort_spools(fetchSpools())

    return render_template('assign_tag.html', spools=spools,
        page_title="NFC")
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))

@app.route("/write_tag")
def write_tag():
  try:
    spool_id = request.args.get("spool_id")

    if not spool_id:
      return render_template('error.html', exception="spool ID is required as a query parameter (e.g., ?spool_id=1)")

    myuuid = str(uuid.uuid4())

    patchExtraTags(spool_id, {}, {
      "tag": json.dumps(myuuid),
    })
    return render_template('write_tag.html', myuuid=myuuid)
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))

@app.route('/', methods=['GET'])
def health():
  return "OK", 200

@app.route("/print_history")
def print_history():
    spoolman_settings = getSettings()

    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 30))
    ref_print_id = request.args.get("ref_print_id", type=int)
    ref_group_id = request.args.get("ref_group_id", type=int)

    filters = {
        "filament_type": request.args.getlist("filament_type"),
        "color": request.args.getlist("color"),
        "filament_id": request.args.getlist("filament_id"),
        "status": request.args.getlist("status")
    }
    if ref_print_id:
        filters["__ref_print_id"] = ref_print_id
    if ref_group_id:
        filters["__ref_group_id"] = ref_group_id
    raw_filament_ids = request.args.get("filament_id", "")
    if raw_filament_ids.strip():
        # Split sur la virgule, supprime espaces et valeurs vides
        filters["filament_id"] = [v.strip() for v in raw_filament_ids.split(",") if v.strip()]
    search = request.args.get("search", "").strip()

    focus_print_id = request.args.get("focus_print_id", type=int)
    focus_group_id = request.args.get("focus_group_id", type=int)

    raw_prints = get_prints_with_filament(filters=filters, search=search)
    spool_list = fetchSpools(False, True)
    spools_by_id = {spool["id"]: spool for spool in spool_list}
    entries = {}

    groups_list = get_print_groups()
    groups_by_id = {g['id']: g for g in groups_list}
    
    all_print_ids = [p["id"] for p in raw_prints]
    tags_by_print = get_tags_for_prints(all_print_ids)
    # ✅ Initialisation safe
    entries = {}
    entries_list = []
    paged_entries = []
    total_pages = 0
    total_prints = 0
    total_duration_seconds = 0
    total_weight = 0
    total_cost = 0
    counts_print = {}
    counts_group = {}
    status_values = []
    focus_group_id = focus_group_id  # (rebind pour clarté)

    for p in raw_prints:
        p["duration"] = float(p.get("duration") or 0.0) / 3600  # pour compatibilité templates
        p["electric_cost"] = p.get("electric_cost", 0.0)
        p["tags"] = get_tags_for_print(p["id"])
        p["translated_name"] = p.get("translated_name", "")
        p["total_price"] = p.get("sold_price_total", 0)
        p["number_of_items"] = p.get("number_of_items", 1)
        p["model_file"] = None

        for filament in p["filament_usage"]:
            if filament.get("spool_id"):
                filament["spool"] = spools_by_id.get(filament["spool_id"])
            filament.setdefault("cost", 0.0)
            filament.setdefault("normal_cost", 0.0)

        if p.get("image_file", "").endswith(".png"):
            model_file = p["image_file"].replace(".png", ".3mf")
            model_path = os.path.join(app.static_folder, 'prints', model_file)
            if os.path.isfile(model_path):
                p["model_file"] = model_file

        if p.get("group_id"):
            gid = p["group_id"]
            entry_key = f"group_{gid}"
            entry = entries.get(entry_key)
            if not entry or entry.get("type") != "group":
                group_data = groups_by_id.get(gid, {})
                entry = {
                    "type": "group",
                    "id": gid,
                    "name": group_data.get("name", f"Groupe {gid}"),
                    "prints": [],
                    "total_duration": 0,
                    "latest_date": p["print_date"],
                    "thumbnail": None,  # image de référence groupe
                    "filament_usage": {},
                    "number_of_items": group_data.get("number_of_items", 1),
                    "primary_print_id": group_data.get("primary_print_id"),
                    "total_cost": group_data.get("total_cost", 0),
                    "electric_cost": group_data.get("electric_cost", 0),
                    "total_normal_cost": group_data.get("total_normal_cost", 0),
                    "total_weight": group_data.get("total_weight", 0),
                    "total_price": group_data.get("sold_price_total", 0),
                    "full_cost": group_data.get("full_cost", 0),
                    "full_normal_cost": group_data.get("full_normal_cost", 0),
                    "full_cost_by_item": group_data.get("full_cost_by_item", 0),
                    "full_normal_cost_by_item": group_data.get("full_normal_cost_by_item", 0),
                    "max_print_id": 0,  # initialisation pour le max print
                    "tags": set(),
                }
                entries[entry_key] = entry
        
            entry["prints"].append(p)
            for t in tags_by_print.get(p["id"], []):
                entry["tags"].add(t)
            entry["total_duration"] += p["duration"]
        
            if parse_print_date(p["print_date"]) > parse_print_date(entry["latest_date"]):
                entry["latest_date"] = p["print_date"]
        
            # Gestion thumbnail selon primary_print_id
            if entry.get("primary_print_id") == p["id"]:
                entry["thumbnail"] = p.get("image_file")
            elif not entry.get("thumbnail") and p.get("image_file"):
                if p["id"] > entry.get("max_print_id", 0):
                    entry["max_print_id"] = p["id"]
                    entry["thumbnail"] = p["image_file"]
        
            for filament in p["filament_usage"]:
                key = filament["spool_id"] or f"{filament['filament_type']}-{filament['color']}"
                if key not in entry["filament_usage"]:
                    entry["filament_usage"][key] = {
                        "grams_used": filament["grams_used"],
                        "cost": filament.get("cost", 0.0),
                        "normal_cost": filament.get("normal_cost", 0.0),
                        "spool": filament.get("spool"),
                        "spool_id": filament.get("spool_id"),
                        "filament_type": filament.get("filament_type"),
                        "color": filament.get("color")
                    }
                else:
                    usage = entry["filament_usage"][key]
                    usage["grams_used"] += filament["grams_used"]
                    usage["cost"] += filament.get("cost", 0.0)
                    usage["normal_cost"] += filament.get("normal_cost", 0.0)


        else:
            entries[f"print_{p['id']}"] = {
                "type": "single",
                "print": p,
                "max_id": p["id"]
            }

    total_prints = sum(
        1 if e["type"] == "single" else len(e.get("prints", []))
        for e in entries.values()
    )
    total_duration_seconds = 0
    total_weight = 0
    total_cost = 0
    
    for entry in entries.values():
        if entry["type"] == "group":
            total_duration_seconds += entry.get("total_duration", 0) * 3600
            total_weight += entry.get("total_weight", 0)
            total_cost += entry.get("full_cost", 0)
        else:
            p = entry["print"]
            total_duration_seconds += (float(p.get("duration") or 0.0) * 3600)
            total_weight += p.get("total_weight", 0)
            total_cost += p.get("full_cost", 0)
    entries_list = sorted(entries.values(), key=lambda e: parse_print_date(e["latest_date"] if e["type"] == "group" else e["print"]["print_date"]), reverse=True)

    all_print_ids = [p["id"] for p in raw_prints]
    counts_print = get_object_counts_by_parent("print", all_print_ids)
    
    group_ids = [e["id"] for e in entries.values() if e["type"] == "group"]
    counts_group = get_object_counts_by_parent("group", group_ids)
    for e in entries.values():
        if e.get("type") == "group" and isinstance(e.get("tags"), set):
            e["tags"] = sorted(e["tags"], key=lambda s: s.lower())
        if e["type"] == "group":
            used = counts_group.get(e["id"], 0)
            e["used_units"] = max(0, int(used))
    
            # Par print du groupe (utile si tu veux aussi le badge “utilisé” par print)
            for p in e.get("prints", []):
                p_used = counts_print.get(p["id"], 0)
                p["used_units"] = max(0, int(p_used))
    
        else:  # single print
            p = e["print"]
            used = counts_print.get(p["id"], 0)
            p["used_units"] = max(0, int(used))
    
        total_pages = (len(entries_list) + per_page - 1) // per_page
        paged_entries = entries_list[(page - 1) * per_page : page * per_page]
    
        if focus_print_id and not focus_group_id:
            for entry in entries_list:
                if entry["type"] == "single" and entry["print"]["id"] == focus_print_id:
                    if entry["print"].get("group_id"):
                        focus_group_id = entry["print"]["group_id"]
                    break
                elif entry["type"] == "group":
                    for p in entry["prints"]:
                        if p["id"] == focus_print_id:
                            focus_group_id = p.get("group_id")
                            break
                    if focus_group_id:
                        break

    distinct_values = get_distinct_values()
    args = request.args.to_dict(flat=False)
    args.pop('page', None)
    groups_list = get_print_groups()
    pagination_pages = compute_pagination_pages(page, total_pages)

    status_values = sorted(set(p.get("status") for p in raw_prints if p.get("status")))

    hours = int(total_duration_seconds // 3600)
    minutes = int((total_duration_seconds % 3600) // 60)
    total_duration_formatted = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

    context = {
        "entries": paged_entries,
        "groups_list": groups_list,
        "currencysymbol": spoolman_settings["currency_symbol"],
        "total_pages": total_pages,
        "filters": filters,
        "distinct_values": distinct_values,
        "page": page,
        "args": filtered_args_for_template(),
        "pagination_pages": pagination_pages,
        "focus_print_id": focus_print_id,
        "focus_group_id": focus_group_id,
        "status_values": status_values,
        "page_title": "Historique",
        "total_prints": total_prints,
        "total_duration_formatted": total_duration_formatted,
        "total_weight": total_weight,
        "total_cost": total_cost
    }
    
    if search:
        context["search"] = search
    
    return render_template("print_history.html", **context)

@app.route("/print_select_spool")
def print_select_spool():

  try:
    ams_slot = request.args.get("ams_slot")
    print_id = request.args.get("print_id")

    if not all([ams_slot, print_id]):
      return render_template('error.html', exception="Missing spool ID or print ID.")

    spools = fetchSpools()
        
    return render_template('print_select_spool.html', spools=spools, ams_slot=ams_slot, print_id=print_id)
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))

@app.route("/history/delete/<int:print_id>", methods=["POST"])
def delete_print_history(print_id):
    data = request.get_json()
    restock = data.get("restock", False)
    ratios = data.get("ratios", {})  # dict {spool_id: ratio_en_%}

    # Récupère la liste des consommations de filament pour ce print
    filament_usages = get_filament_for_print(print_id)

    if restock:
        for usage in filament_usages:
            spool_id = usage["spool_id"]
            grams_used = usage["grams_used"]

            if spool_id:
                # Si un ratio est fourni pour ce spool_id, sinon 100%
                ratio_percent = ratios.get(str(spool_id), 100)
                ratio = max(0, min(100, ratio_percent)) / 100.0
                adjusted_grams = grams_used * ratio

                # Remet en stock la quantité ajustée
                consumeSpool(spool_id, -adjusted_grams)

    # Supprime le print et ses usages
    delete_print(print_id)

    return jsonify({"status": "ok"})

@app.route("/history/reajust/<int:print_id>", methods=["POST"])
def reajust_print_history(print_id):
    data = request.get_json()
    ratios = data.get("ratios", {})

    filament_usages = get_filament_for_print(print_id)

    for usage in filament_usages:
        spool_id = usage["spool_id"]
        grams_used = usage["grams_used"]

        if spool_id:
            ratio_percent = ratios.get(str(spool_id), 100)
            ratio = max(0, min(100, ratio_percent)) / 100.0
            adjusted_grams = grams_used * ratio

            consumeSpool(spool_id, -adjusted_grams)

            # 🔷 Met à jour filament_usage
            new_grams_used = grams_used - adjusted_grams
            update_filament_usage(print_id, spool_id, new_grams_used)

    return jsonify({"status": "ok"})

@app.route("/history/<int:print_id>/filaments", methods=["GET"])
def get_print_filaments(print_id):
    filament_usages = get_filament_for_print(print_id)

    enriched = []
    for usage in filament_usages:
        spool_id = usage["spool_id"]
        grams_used = usage["grams_used"]

        spool = getSpoolById(spool_id) if spool_id else None
        if spool:
            color = spool.get("filament", {}).get("color_hex")
            vendor = spool.get("filament", {}).get("vendor", {}).get("name", "UnknownVendor")
            material = spool.get("filament", {}).get("material", "UnknownMaterial")
            realName = spool.get("filament", {}).get("name", "UnknownName")
            name = f"#{spool_id} - {realName} - {vendor} - {material}"
        else:
            color = usage.get("color", "#000000")
            name = usage.get("filament_type", "N/A")
        if color:
            color = f"#{color}"
        enriched.append({
            "spool_id": spool_id,
            "grams_used": grams_used,
            "name": name,
            "color": color
        })

    return jsonify(enriched)

@app.route("/history/<int:print_id>/tags", methods=["GET"])
def get_tags(print_id):
    tags = get_tags_for_print(print_id)
    return jsonify(tags)

@app.route("/history/<int:print_id>/tags/add", methods=["POST"])
def add_tag(print_id):
    tag_input = request.form.get("tag", "")
    # ✂️ Découpe sur les virgules ou points-virgules
    tags = [t.strip() for t in re.split(r'[;,]', tag_input) if t.strip()]
    for tag in tags:
        add_tag_to_print(print_id, tag)
    return jsonify({"status": "ok"})

@app.route("/history/<int:print_id>/tags/remove", methods=["POST"])
def remove_tag(print_id):
    tag = request.form.get("tag")
    if tag:
        remove_tag_from_print(print_id, tag)
    return jsonify({"status": "ok"})
    
@app.route("/history/group/<int:group_id>/tags", methods=["GET"])
def get_group_tags(group_id):
    return jsonify(get_tags_for_group(group_id))

@app.route("/history/group/<int:group_id>/tags/add", methods=["POST"])
def add_group_tag(group_id):
    tag_input = request.form.get("tag", "")
    tags = [t.strip() for t in re.split(r'[;,]', tag_input) if t.strip()]
    for tag in tags:
        add_tag_to_group(group_id, tag)
    return redirect_with_context("print_history", focus_group_id=group_id)

@app.route("/history/group/<int:group_id>/tags/remove", methods=["POST"])
def remove_group_tag(group_id):
    tag = request.form.get("tag", "")
    if tag:
        remove_tag_from_group(group_id, tag)
    return redirect_with_context("print_history", focus_group_id=group_id)
    
@app.route("/filaments")
def filaments():
    #if not isMqttClientConnected():
    #    return render_template('error.html', exception="L'imprimante est elle allumée ? Avez vous renseigné les paramètres ?")

    ams_id = request.args.get("ams")
    tray_id = request.args.get("tray")
    spool_id = request.args.get("spool_id")
    # 🎯 Si un spool est sélectionné en mode fill : action directe
    if spool_id and ams_id and tray_id:
        spool_data = getSpoolById(spool_id)
        tray_uuid = request.args.get("tray_uuid")
        tray_info_idx = request.args.get("tray_info_idx")
        tray_color = request.args.get("tray_color")
        if tray_uuid and tray_info_idx and tray_color:
            set_tray_spool_map(tray_uuid, tray_info_idx, tray_color, spool_id)
        setActiveTray(spool_id, spool_data["extra"], ams_id, tray_id)
        setActiveSpool(ams_id, tray_id, spool_data)
        return redirect(url_for('home', success_message=f"La bobine {spool_id} a été assigné à l'emplacement {tray_id} de l'AMS {ams_id}."))

    # 🔁 Sinon, affichage des bobines filtrées/paginées
    page = int(request.args.get("page", 1))
    per_page = 25
    search = request.args.get("search", "").lower()
    sort = request.args.get("sort", "default")
    selected_family = request.args.get("color")
    include_archived = request.args.get("include_archived") == "1"

    assign_print_id = request.args.get("assign_print_id")
    assign_filament_index = request.args.get("assign_filament_index")
    assign_page = request.args.get("assign_page")
    assign_search = request.args.get("assign_search")
    filament_usage = request.args.get("filament_usage", '0')

    is_assign_mode = all([assign_print_id, assign_filament_index])
    is_fill_mode = all([ams_id, tray_id])
    tray_uuid = tray_info_idx = tray_color = None
    if is_fill_mode:
        tray_uuid = request.args.get("tray_uuid")
        tray_info_idx = request.args.get("tray_info_idx")
        tray_color = request.args.get("tray_color")
    all_filaments = fetchSpools(False, include_archived) or []

    if search:
        search_terms = search.split()
        def matches(f):
            filament = f.get("filament", {})
            vendor = filament.get("vendor", {})
            fields = [
                filament.get("name", "").lower(),
                filament.get("material", "").lower(),
                vendor.get("name", "").lower(),
                f.get("location", "").lower(),
            ]
            return all(any(term in field for field in fields) for term in search_terms)
        all_filaments = [f for f in all_filaments if matches(f)]

    all_families_in_page = set()
    for spool in all_filaments:
        filament = spool.get("filament", {})
        hexes = []
        if filament.get("multi_color_hexes"):
            hexes = filament["multi_color_hexes"].split(",") if isinstance(filament["multi_color_hexes"], str) else filament["multi_color_hexes"]
        elif filament.get("color_hex"):
            hexes = [filament["color_hex"]]
        families = set()
        for hx in hexes:
            fams = two_closest_families(hx, threshold=60)
            families.update(fams)
        spool["color_families"] = sorted(families)
        all_families_in_page.update(families)

    if selected_family:
        all_filaments = [
            f for f in all_filaments if selected_family in f.get("color_families", [])
        ]

    def sort_key(f):
        filament = f.get("filament", {})
        vendor = filament.get("vendor", {})
        return (
            f.get("location", "").lower(),
            filament.get("material", "").lower(),
            vendor.get("name", "").lower(),
            filament.get("name", "").lower()
        )

    if sort == "remaining":
        all_filaments.sort(key=lambda f: f.get("remaining_weight") or 0)
    else:
        all_filaments.sort(key=sort_key)

    total = len(all_filaments)
    total_pages = math.ceil(total / per_page)
    filaments_page = all_filaments[(page - 1) * per_page: page * per_page]

    total_remaining = sum(f.get("remaining_weight") or 0 for f in all_filaments)
    vendor_names = {
        f.get("filament", {}).get("vendor", {}).get("name", "")
        for f in all_filaments if f.get("filament", {}).get("vendor")
    }
    total_vendors = len(vendor_names)

    return render_template(
        "filaments.html",
        filaments=filaments_page,
        page=page,
        total_pages=total_pages,
        search=search,
        sort=sort,
        all_families=sorted(all_families_in_page),
        selected_family=selected_family,
        include_archived=include_archived,
        assign_print_id=assign_print_id,
        assign_filament_index=assign_filament_index,
        assign_page=assign_page,
        assign_search=assign_search,
        is_assign_mode=is_assign_mode,
        ams_id=ams_id,
        tray_id=tray_id,
        filament_usage=filament_usage,
        total_filaments=total,
        total_vendors=total_vendors,
        total_remaining=total_remaining,
        page_title="Filaments",
        tray_uuid=tray_uuid,
        tray_info_idx=tray_info_idx,
        tray_color=tray_color,
        args=_merge_context_args()
    )


@app.route("/edit_print_name", methods=["POST"])
def edit_print_name():
    print_id = int(request.form["print_id"])
    new_filename = request.form.get("file_name", "").strip()

    if new_filename:
        update_print_filename(print_id, new_filename)
    return redirect_with_context("print_history", focus_print_id=print_id)


@app.route("/edit_print_items", methods=["POST"])
def edit_print_items():
    print_id = int(request.form["print_id"])
    try:
        number_of_items = int(request.form["number_of_items"])
        if number_of_items < 1:
            number_of_items = 1
    except (ValueError, TypeError):
        number_of_items = 1

    update_print_history_field(print_id, "number_of_items", number_of_items)
    return redirect_with_context("print_history", focus_print_id=print_id)


@app.route("/create_group", methods=["POST"])
def create_group():
    print_id = int(request.form["print_id"])
    group_name = request.form["group_name"].strip()

    if group_name:
        group_id = create_print_group(group_name)
        update_print_history_field(print_id, "group_id", group_id)
        update_group_created_at(group_id)
        return redirect_with_context(
            "print_history",
            focus_print_id=print_id,
            focus_group_id=group_id
        )

    return redirect_with_context("print_history", focus_print_id=print_id)


@app.route("/assign_to_group", methods=["POST"])
def assign_to_group():
    group_id_or_name = request.form["group_id_or_name"]
    print_id = int(request.form["print_id"])

    if group_id_or_name.isdigit():
        group_id = int(group_id_or_name)
    else:
        group_id = create_print_group(group_id_or_name)

    update_print_history_field(print_id, "group_id", group_id)
    update_group_created_at(group_id)

    return redirect_with_context(
        "print_history",
        focus_print_id=print_id,
        focus_group_id=group_id
    )


@app.route("/remove_from_group", methods=["POST"])
def remove_from_group():
    print_id = int(request.form["print_id"])
    group_id = get_group_id_of_print(print_id)

    update_print_history_field(print_id, "group_id", None)
    if group_id:
        update_group_created_at(group_id)

    return redirect_with_context(
        "print_history",
        focus_print_id=print_id
    )


@app.route("/rename_group", methods=["POST"])
def rename_group():
    group_id = int(request.form["group_id"])
    group_name = request.form["group_name"].strip()

    if group_name:
        update_print_group_field(group_id, "name", group_name)
    
    return redirect_with_context("print_history",focus_group_id=group_id)


@app.route("/edit_group_items", methods=["POST"])
def edit_group_items():
    group_id = int(request.form["group_id"])
    try:
        number_of_items = int(request.form["number_of_items"])
        if number_of_items < 1:
            number_of_items = 1
    except (ValueError, TypeError):
        number_of_items = 1

    update_print_group_field(group_id, "number_of_items", number_of_items)

    return redirect_with_context("print_history",focus_group_id=group_id)

@app.route("/api/groups/search")
def api_groups_search():
    q = request.args.get("q", "").strip()
    groups = get_print_groups()
    results = []
    for group in groups:
        if q.lower() in group["name"].lower():
            created_at_str = group.get("created_at")
            try:
                created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                created_at_fmt = created_at.strftime('%d/%m/%Y %H:%M')
            except Exception:
                created_at_fmt = created_at_str or "?"
            label = f"{group['name']} (créé le {created_at_fmt})"
            results.append({"id": group["id"], "text": label})
    return jsonify({"results": results})
    
@app.route('/spool/<int:spool_id>/reajust', methods=['POST'])
def reajust_spool_route(spool_id):
    try:
        new_weight = float(request.form.get('new_weight'))
    except (ValueError, TypeError):
        flash('Poids invalide', 'danger')
        return redirect(request.referrer or url_for('filament_page'))

    response = reajust_spool(spool_id, new_weight)
    return redirect(request.referrer or url_for('filament_page'))

@app.route('/spool/<int:spool_id>/archive', methods=['POST'])
def archive_spool_route(spool_id):
    response = archive_spool(spool_id)
    return redirect(request.referrer or url_for('filament_page'))

@app.route("/download_model/<filename>")
def download_model(filename):
    # Sécuriser le nom du fichier réel (dans 'static/prints')
    safe_filename = os.path.basename(filename)

    # Nom souhaité côté utilisateur (dans l'URL `?as=...`)
    requested_name = request.args.get("as", safe_filename)

    # Nettoyage du nom pour éviter les injections ou caractères suspects
    cleaned_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', requested_name)
    if not cleaned_name.endswith('.3mf'):
        cleaned_name += '.3mf'

    # Vérifie que le fichier existe avant d’envoyer
    full_path = os.path.join('static', 'prints', safe_filename)
    if not os.path.isfile(full_path):
        return abort(404)

    return send_from_directory(
        os.path.join('static', 'prints'),
        safe_filename,
        as_attachment=True,
        download_name=cleaned_name
    )

@app.route("/stats")
def stats():
    from print_history import get_distinct_values

    spoolman_settings = getSettings()

    # Récupérer les filtres, recherche et période
    filters = {
        "filament_type": request.args.getlist("filament_type"),
        "color": request.args.getlist("color")
    }
    search = request.args.get("search", "").strip()
    period = request.args.get("period", "all")

    # Calcul des stats avec filtres
    stats_data = get_statistics(period=period, filters=filters, search=search)

    return render_template(
        "stats.html",
        stats=stats_data,
        currencysymbol=spoolman_settings["currency_symbol"],
        selected_period=period,
        filters=filters,
        search=search,
        distinct_values=get_distinct_values(),
        page_title="Statistiques",
        args=_merge_context_args()
    )

@app.route("/adjust_duration", methods=["POST"])
def adjust_duration():
    print_id = int(request.form["print_id"])

    try:
        hours = float(request.form.get("hours", 0) or 0)
        minutes = float(request.form.get("minutes", 0) or 0)
        total_seconds = int((hours * 60 + minutes) * 60)
        adjustDuration(print_id, total_seconds)
    except Exception:
        pass
    
    return redirect_with_context("print_history",focus_print_id=print_id)


@app.route("/set_group_primary", methods=["POST"])
def set_group_primary():
    print_id = int(request.form["print_id"])
    group_id = int(request.form["group_id"])
    set_group_primary_print(group_id, print_id)
    return redirect_with_context("print_history",focus_print_id=print_id,focus_group_id=group_id)

@app.route('/assign_spool_to_print', methods=['POST'])
def assign_spool_to_print():
    spool_id = int(request.form['spool_id'])
    print_id = int(request.form['print_id'])
    filament_index = int(request.form['filament_index'])
    update_filament_spool(print_id=print_id, filament_id=filament_index, spool_id=spool_id)
    skip_usage = request.form.get("skip_usage") == "1"
    if not skip_usage:
        consumeSpool(spool_id, float(request.form.get("filament_usage") or 0))
        
    return redirect_with_context(
       "print_history",
        focus_print_id=print_id
    )


@app.route("/change_print_status", methods=["POST"])
def change_print_status():
    print_id = int(request.form.get("print_id"))
    new_status = request.form.get("new_status", "SUCCESS").strip()
    note = request.form.get("status_note", "").strip()

    if new_status not in {"SUCCESS", "IN_PROGRESS", "FAILED", "PARTIAL", "TO_REDO"}:
        return redirect_with_context("print_history",focus_print_id=print_id)

    update_print_history_field(print_id, "status", new_status)
    update_print_history_field(print_id, "status_note", note)
    return redirect_with_context("print_history",focus_print_id=print_id)

@app.route("/set_sold_price", methods=["POST"])
def set_sold_price():
    item_id = int(request.form.get("id"))
    is_group = bool(int(request.form.get("is_group", 0)))
    total_price = float(request.form.get("total_price") or 0)
    sold_units = int(request.form.get("sold_units") or 0)

    if item_id <= 0:
        return redirect(request.referrer or url_for("print_history"))

    set_sold_info(print_id=item_id, is_group=is_group, total_price=total_price, sold_units=sold_units)

    if is_group:
        return redirect_with_context("print_history",focus_group_id=item_id)
    else:
        return redirect_with_context("print_history",focus_print_id=item_id)

@app.route("/admin/manual_print", methods=["POST"])
def admin_manual_print():
    try:
        files = request.files.getlist("file[]")
        print_datetime = request.form.get("datetime")

        if not files or all(f.filename == "" for f in files):
            flash("Veuillez sélectionner au moins un fichier .3MF.", "danger")
            return redirect(url_for("auth.settings"))

        # Validation de la date
        try:
            try:
                custom_datetime = datetime.strptime(print_datetime, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                custom_datetime = datetime.strptime(print_datetime, "%Y-%m-%dT%H:%M")
        except Exception:
            flash("Format de date invalide.", "danger")
            return redirect(url_for("auth.settings"))

        upload_dir = os.path.join(current_app.root_path, "data", "temp_uploads")
        os.makedirs(upload_dir, exist_ok=True)

        successes = []
        errors = []

        for file in files:
            if not file or not file.filename.lower().endswith(".3mf"):
                errors.append(f"{file.filename} : format invalide")
                continue

            try:
                filename = secure_filename(file.filename)
                temp_path = os.path.join(upload_dir, filename)
                file.save(temp_path)

                result = insert_manual_print(temp_path, custom_datetime)
                if "error" in result:
                    errors.append(f"{filename} : {result['error']}")
                else:
                    successes.append(f"{filename} (ID #{result['print_id']})")

            except Exception as e:
                traceback.print_exc()
                errors.append(f"{file.filename} : {str(e)}")

        if successes:
            flash(f"{len(successes)} impression(s) ajoutée(s) : " + ", ".join(successes), "success")
        if errors:
            flash(f"⚠ Erreurs : " + "; ".join(errors), "danger")

        return redirect(url_for("auth.settings"))

    except Exception as e:
        traceback.print_exc()
        flash(f"Erreur serveur : {str(e)}", "danger")
        return redirect(url_for("auth.settings"))

@app.route("/recalculate_all_costs", methods=["POST"])
def recalculate_all_costs():

    start_time = time.time()

    spools = fetchSpools(cached=False,archived=True)
    spools_by_id = {spool["id"]: spool for spool in spools}

    all_prints = get_prints_with_filament()
    for p in all_prints:
        recalculate_print_data(p["id"], spools_by_id)

    groups = get_print_groups()
    for group in groups:
        recalculate_group_data(group["id"], spools_by_id)

    duration = time.time() - start_time
    flash(f"✅ Tous les coûts ont été recalculés en {duration:.2f} secondes.")
    return redirect(url_for("auth.settings"))

@app.route("/cleanup_orphans", methods=["POST"])
def cleanup_orphans():
    cleanup_orphan_data()
    flash("🧹 Données orphelines supprimées avec succès.", "success")
    return redirect(url_for("auth.settings"))
    
@app.route('/printer_status')
def api_printer_status():
    """
    Renvoie l'état complet pour le rafraîchissement JS:
    - PRINTER_STATUS enrichi (progress, status, ETA, temps restants, températures…)
    - ams_data (liste des AMS avec leurs trays)
    - vt_tray_data (spool externe)
    - issue (bool pour signalisation)
    - printName + thumbnail (comme sur la home)
    """
    try:
        # Copie snapshot de l'état imprimante
        with PRINTER_STATUS_LOCK:
            status_copy = dict(PRINTER_STATUS)

        # ---- Récupération/augmentation des données AMS & trays (idem home) ----
        last_ams_config = getLastAMSConfig()
        ams_data = last_ams_config.get("ams", []) or []
        vt_tray_data = last_ams_config.get("vt_tray", {}) or {}

        spool_list = fetchSpools()

        # External spool
        augmentTrayDataWithSpoolMan(spool_list, vt_tray_data, trayUid(EXTERNAL_SPOOL_AMS_ID, EXTERNAL_SPOOL_ID))
        issue = bool(vt_tray_data.get("issue"))

        # Trays AMS
        for ams in ams_data:
            for tray in ams.get("tray", []) or []:
                augmentTrayDataWithSpoolMan(spool_list, tray, trayUid(ams["id"], tray["id"]))
                issue |= bool(tray.get("issue"))

        # Locations (si configurées)
        LOCATION_MAPPING = get_app_setting("AMS_LOCATION_MAPPING", "")
        if LOCATION_MAPPING:
            d = dict(item.split(":") for item in LOCATION_MAPPING.split(";") if ":" in item)
            for ams in ams_data:
                ams_name = f'AMS_{ams["id"]}'
                ams['location'] = d.get(ams_name)

        # Ordre des AMS
        AMS_ORDER = get_app_setting("AMS_ORDER", "")
        if AMS_ORDER:
            mapping = {int(k): int(v) for k, v in (item.split(":") for item in AMS_ORDER.split(";") if ":" in item)}
            reordered = [None] * len(ams_data)
            for src_index, dst_index in mapping.items():
                if 0 <= src_index < len(ams_data) and 0 <= dst_index < len(ams_data):
                    reordered[dst_index] = ams_data[src_index]
            # Remplir les trous éventuels en conservant l'ordre original
            fallback = [a for a in ams_data if a not in reordered]
            for i in range(len(reordered)):
                if reordered[i] is None and fallback:
                    reordered[i] = fallback.pop(0)
            ams_data = reordered

        # Dernière impression (nom + vignette)
        latest = get_latest_print()
        if latest:
            status_copy["printName"] = latest.get("file_name")
            status_copy["thumbnail"] = latest.get("image_file")
        else:
            status_copy["printName"] = None
            status_copy["thumbnail"] = None

        # Réponse JSON complète
        payload = dict(status_copy)
        payload.update({
            "ams_data": ams_data,
            "vt_tray_data": vt_tray_data,
            "issue": issue,
        })
        return jsonify(payload)
    except Exception as e:
        current_app.logger.exception("printer_status failed")
        return jsonify({"error": str(e)}), 500
        
@app.route("/tray_mappings")
def tray_mappings():
    mappings = get_all_tray_spool_mappings()
    return jsonify(mappings)

@app.route("/clear_tray_mappings", methods=["POST"])
def clear_tray_mappings():
    delete_all_tray_spool_mappings()
    flash("Tous les mappings ont été supprimés.", "success")
    return redirect(url_for("auth.settings"))

@app.route("/api/objects/available")
def api_objects_available():
    source_type = request.args.get("type", "").strip()
    if source_type not in ("print", "group"):
        return jsonify({"available": 0}), 400
    source_id = int(request.args.get("id", "0") or 0)
    return jsonify({"available": get_available_units(source_type, source_id)})

@app.route("/api/objects/create", methods=["POST"])
def api_objects_create():
    source_type = request.form.get("type", "").strip()
    source_id = int(request.form.get("id", "0"))
    qty = int(request.form.get("qty", "0"))
    created = create_objects_from_source(source_type, source_id, qty)
    return jsonify({"status": "ok", "created": created})

@app.route("/objects")
def objects_page():
    page = int(request.args.get("page", "1") or 1)
    filters = {
        "search": request.args.get("search", ""),
        "source_type": request.args.get("source_type", ""),  # print | group | ''
        "sale_filter": request.args.get("sale_filter", ""),  # '' | vendus | dispo | offert
    }
    rows, total_pages = list_objects(filters, page, per_page=30)

    # NEW: agrégats pour les tuiles
    summary = summarize_objects(filters)

    # Tags des objets (batch)
    ids = [r["id"] for r in rows]
    obj_tags = get_tags_for_objects(ids) if ids else {}
    obj_accessories = {oid: list_object_accessories(oid) for oid in ids} if ids else {}

    # Devise (comme print_history)
    spoolman_settings = getSettings(cached=True)

    return render_template(
        "objects.html",
        objects=rows,
        page=page,
        total_pages=total_pages,
        filters=filters,
        args=request.args,
        obj_tags=obj_tags,
        obj_accessories=obj_accessories,
        page_title="Objets",
        currencysymbol=spoolman_settings.get("currency_symbol", "€"),
        # expose les agrégats
        total_objects=summary["total_objects"],
        sold_count=summary["sold_count"],
        available_count=summary["available_count"],
        gifted_count=summary["gifted_count"],
        personal_count=summary["personal_count"],
        sum_sold_price=summary["sum_sold_price"],
        sum_positive_margin=summary["sum_positive_margin"],
    )


@app.post("/objects/<int:object_id>/rename")
def objects_rename(object_id):
    new_name = (request.form.get("name") or "").strip()
    if new_name:
        rename_object(object_id, new_name)
    return redirect(url_for("objects_page", **request.args.to_dict(flat=True)))

@app.post("/objects/<int:object_id>/delete")
def objects_delete(object_id):
    delete_object(object_id)
    return redirect(url_for("objects_page", **request.args.to_dict(flat=True)))

@app.route("/objects/<int:object_id>/sell", methods=["POST"])
def objects_sell(object_id: int):
    raw_price = (request.form.get("sold_price") or "").strip()
    raw_date  = (request.form.get("sold_date") or "").strip()
    comment   = (request.form.get("sold_comment") or "").strip() or None

    # --- TRACE pour vérifier que la route est bien appelée
    logger.debug(f"[objects_sell] object_id={object_id} price='{raw_price}' date='{raw_date}'")

    # Prix : vide => 0 (don)
    try:
        price = 0.0 if raw_price == "" else float(raw_price)
    except ValueError:
        flash("Prix invalide. Utilisez un nombre (0 pour un don).", "danger")
        return redirect_with_context("objects_page", focus_object_id=object_id)

    if price < 0:
        flash("Le prix ne peut pas être négatif.", "danger")
        return redirect_with_context("objects_page", focus_object_id=object_id)

    sold_personal = False
    if price == 0:
        sold_personal = (request.form.get("sold_personal") == "1")
    sold_date = raw_date or date.today().isoformat()

    try:
        update_object_sale(
            object_id=object_id,
            sold_price=price,
            sold_date=sold_date,
            comment=comment,
            sold_personal=sold_personal,
        )
    except Exception as e:
        logger.error(f"[objects_sell][ERROR] {e}")
        flash(f"Échec d'enregistrement de la vente : {e}", "danger")
        return redirect_with_context("objects_page", focus_object_id=object_id)

    flash("Vente / don enregistré.", "success")
    return redirect_with_context("objects_page", focus_object_id=object_id)

@app.route("/objects/<int:object_id>/unsell", methods=["POST"])
def objects_unsell(object_id: int):
    try:
        clear_object_sale(object_id)
    except Exception as e:
        # log(...) si tu as un logger
        flash(f"Échec de l'annulation de la vente : {e}", "danger")
        return redirect_with_context("objects_page", focus_object_id=object_id)

    flash("Vente annulée.", "success")
    return redirect_with_context("objects_page", focus_object_id=object_id)

@app.route("/objects/<int:object_id>/comment", methods=["POST"])
def objects_update_comment(object_id: int):
    comment = (request.form.get("comment") or "").strip()
    # None = on efface ; chaîne non vide = on enregistre
    try:
        update_object_comment(object_id, comment if comment else None)
    except Exception as e:
        flash(f"Échec de la mise à jour du commentaire : {e}", "danger")
        return redirect_with_context("objects_page", focus_object_id=object_id)

    flash("Commentaire mis à jour.", "success")
    return redirect_with_context("objects_page", focus_object_id=object_id)

@app.route("/objects/<int:object_id>/tags/add", methods=["POST"])
def add_object_tag(object_id: int):
    tag_input = request.form.get("tag", "")
    tags = [t.strip() for t in re.split(r"[;,]", tag_input) if t.strip()]
    for tag in tags:
        add_object_tag(object_id, tag)

    tags_now = get_tags_for_objects([object_id]).get(object_id, [])
    return jsonify({"status": "ok", "tags": tags_now})


@app.route("/objects/<int:object_id>/tags/remove", methods=["POST"])
def remove_object_tag(object_id: int):
    tag = (request.form.get("tag", "")).strip()
    if tag:
        remove_object_tag(object_id, tag)
    tags_now = get_tags_for_objects([object_id]).get(object_id, [])
    return jsonify({"status": "ok", "tags": tags_now})

@app.route("/accessories")
def accessories_list():
    q     = (request.args.get("q") or "").strip()
    sort  = (request.args.get("sort") or "").lower()
    order = (request.args.get("order") or "asc").lower()
    order = "desc" if order == "desc" else "asc"

    rows = list_accessories()  # ← ta fonction existante

    # Filtre nom (contient, insensible à la casse)
    if q:
        lq = q.lower()
        rows = [r for r in rows if lq in (r.get("name") or "").lower()]

    # Tri
    if sort == "qty":
        rows.sort(key=lambda r: (r.get("quantity") or 0), reverse=(order == "desc"))
    elif sort == "price":
        rows.sort(key=lambda r: (r.get("unit_price") or 0.0), reverse=(order == "desc"))
    else:
        # tri par nom par défaut
        rows.sort(key=lambda r: (r.get("name") or "").lower())

    return render_template(
        "accessories.html",
        accessories=rows,
        args=request.args,
        filters={"q": q, "sort": sort, "order": order},
        currencysymbol=current_app.config.get("CURRENCY_SYMBOL", "€"),
        page_title="Accessoires",  # pour le bouton contextuel dans base.html si tu l’utilises
    )
    
@app.route("/accessories/add", methods=["POST"])
def accessories_add():
    name = (request.form.get("name") or "").strip()
    qty = int(request.form.get("qty") or 0)
    total_price = float(request.form.get("total_price") or 0)

    image_path = None
    file = request.files.get("image")
    if file and file.filename and _allowed_image(file.filename):
        fname = secure_filename(file.filename)
        # Préfixe simple pour éviter collisions
        final_name = f"{name.replace(' ', '_')}_{fname}"
        save_path = os.path.join(ACCESSORY_UPLOAD_DIR, final_name)
        file.save(save_path)
        # Chemin relatif à /static
        image_path = f"uploads/accessories/{final_name}"

    try:
        create_accessory(name=name, qty=qty, total_price=total_price, image_path=image_path)
        flash("Accessoire ajouté avec succès.", "success")
    except Exception as e:
        flash(f"Erreur à la création de l’accessoire : {e}", "danger")

    return redirect(url_for("accessories_list"))

@app.route("/accessories/add_stock", methods=["POST"])
def accessories_add_stock_route():
    acc_id = int(request.form.get("acc_id") or 0)
    add_qty = int(request.form.get("add_qty") or 0)
    add_total_price = float(request.form.get("add_total_price") or 0)

    try:
        add_accessory_stock(acc_id=acc_id, add_qty=add_qty, add_total_price=add_total_price)
        flash("Stock ajouté et prix unitaire mis à jour.", "success")
    except Exception as e:
        flash(f"Erreur à l’ajout de stock : {e}", "danger")

    return redirect(url_for("accessories_list"))
    
@app.route("/accessories/remove_stock", methods=["POST"])
def accessories_remove_stock_route():
    acc_id = int(request.form.get("acc_id") or 0)
    remove_qty = int(request.form.get("remove_qty") or 0)

    try:
        remove_accessory_stock(acc_id=acc_id, remove_qty=remove_qty)
        flash("Stock retiré.", "success")
    except Exception as e:
        flash(f"Erreur au retrait de stock : {e}", "danger")

    return redirect(url_for("accessories_list"))

@app.route("/api/accessories/search")
def api_accessories_search():
    q = (request.args.get("q") or "").strip().lower()

    # On liste puis on filtre ici côté API pour ne renvoyer que les accessoires avec stock > 0
    rows = [r for r in list_accessories() if (r.get("quantity") or 0) > 0]

    if q:
        rows = [r for r in rows if q in (r.get("name") or "").lower()]

    rows = rows[:30]

    def as_item(r):
        # image_path est stocké en relatif (ex: 'uploads/accessories/acc_12_xxx.png')
        img_url = url_for("static", filename=r["image_path"]) if r.get("image_path") \
                  else url_for("static", filename="placeholder.png")
        return {
            "id": r["id"],
            "name": r["name"],
            "quantity": r["quantity"],
            "unit_price": r["unit_price"],
            "image_url": img_url,
        }

    return jsonify([as_item(r) for r in rows])

@app.route("/api/accessories/<int:acc_id>/usage")
def api_accessory_usage(acc_id: int):
    # Renvoie les objets liés à cet accessoire
    rows = list_objects_using_accessory(acc_id)  # fonction à ajouter dans objects.py si absente
    # Normaliser thumbnail -> URL absolue si besoin
    items = []
    for r in rows:
        thumb = r.get("thumbnail") or r.get("source_thumbnail")
        if thumb and not thumb.startswith("/"):
            thumb = url_for("static", filename=thumb)
        items.append({
            "object_id": r["object_id"],
            "object_name": r["object_name"],
            "quantity": r.get("quantity") or 0,             # qté de cet accessoire dans l’objet
            "thumbnail": thumb,
            "parent_type": r.get("parent_type"),            # "print" ou "group"
            "parent_id": r.get("parent_id"),
        })
    return jsonify(items)

@app.route("/objects/<int:object_id>/add_accessory", methods=["POST"])
def objects_add_accessory(object_id: int):
    accessory_id = int(request.form.get("accessory_id") or 0)
    qty = int(request.form.get("qty") or 0)
    try:
        link_accessory_to_object(object_id, accessory_id, qty)
        flash("Accessoire ajouté à l’objet.", "success")
    except Exception as e:
        flash(f"Erreur lors de l’ajout d’accessoire : {e}", "danger")
    return redirect_with_context("objects_page", focus_object_id=object_id)

@app.route("/objects/<int:object_id>/remove_accessory", methods=["POST"])
def objects_remove_accessory(object_id: int):
    accessory_id = int(request.form.get("accessory_id") or 0)
    qty = request.form.get("qty")
    qty = int(qty) if qty not in (None, "",) else None
    try:
        unlink_accessory_from_object(object_id, accessory_id, qty=qty)
        flash("Accessoire retiré de l’objet.", "success")
    except Exception as e:
        flash(f"Erreur lors du retrait d’accessoire : {e}", "danger")
    return redirect_with_context("objects_page", focus_object_id=object_id)

@app.route("/accessories/<int:acc_id>/delete", methods=["POST"])
def accessories_delete_route(acc_id: int):
    acc = get_accessory(acc_id)
    if not acc:
        flash("Accessoire introuvable.", "danger")
        return redirect(url_for("accessories_list"))

    old_rel_path = acc.get("image_path")
    try:
        delete_accessory(acc_id)
        # On tente de supprimer l'ancien fichier visuel
        if old_rel_path:
            old_abs = _abs_static_path(old_rel_path)
            try:
                if os.path.exists(old_abs):
                    os.remove(old_abs)
            except Exception as fe:
                app.logger.warning(f"Échec suppression visuel après delete: {old_abs} ({fe})")

        flash("Accessoire supprimé.", "success")
    except Exception as e:
        flash(f"Erreur lors de la suppression : {e}", "danger")

    return redirect(url_for("accessories_list"))

@app.route("/accessories/<int:acc_id>/upload_image", methods=["POST"])
def accessories_upload_image(acc_id: int):
    file = request.files.get("image")
    if not (file and file.filename and _allowed_image(file.filename)):
        flash("Fichier d’image invalide.", "warning")
        return redirect(url_for("accessories_list"))

    acc = get_accessory(acc_id)
    if not acc:
        flash("Accessoire introuvable.", "danger")
        return redirect(url_for("accessories_list"))

    # Sauvegarde du nouveau fichier
    fname = secure_filename(file.filename)
    final_name = f"acc_{acc_id}_{fname}"
    save_path = os.path.join(ACCESSORY_UPLOAD_DIR, final_name)
    file.save(save_path)
    new_rel_path = f"uploads/accessories/{final_name}"

    old_rel_path = acc.get("image_path")

    try:
        # MAJ BDD (DAL)
        set_accessory_image_path(acc_id, new_rel_path)
        # Suppression de l'ancien fichier si différent
        if old_rel_path and old_rel_path != new_rel_path:
            old_abs = _abs_static_path(old_rel_path)
            try:
                if os.path.exists(old_abs):
                    os.remove(old_abs)
            except Exception as fe:
                # On ne bloque pas pour un échec de suppression disque
                app.logger.warning(f"Échec suppression ancien visuel: {old_abs} ({fe})")

        flash("Visuel mis à jour.", "success")
    except Exception as e:
        flash(f"Erreur lors de la mise à jour du visuel : {e}", "danger")

    return redirect(url_for("accessories_list"))

@app.post("/accessories/<int:acc_id>/rename")
def accessories_rename_route(acc_id: int):
    new_name = request.form.get("name", "").strip()
    if not new_name:
        flash("Nom requis", "danger")
        return redirect_with_context("accessories_list")

    rename_accessory(acc_id, new_name)  # fonction à créer dans accessories.py
    flash(f"Accessoire renommé en {new_name}", "success")
    return redirect_with_context("accessories_list", focus_acc_id=acc_id)


app.register_blueprint(auth_bp)