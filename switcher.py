# switcher.py
from flask import Blueprint, redirect, abort, request
from flask_login import login_required
from installations import get_installation, _normalize_guest_url  # <— on réutilise la même
from config import get_app_setting
from urllib.parse import urlsplit, urlunsplit

switch_bp = Blueprint('switch', __name__)

def _current_origin() -> str:
    base = get_app_setting("BASE_URL", "").strip()
    if base:
        return base if base.endswith('/') else base + '/'
    scheme = "https" if request.headers.get("X-Forwarded-Proto","").lower()=="https" or request.is_secure else "http"
    return f"{scheme}://{request.host}/"

@switch_bp.route("/switch/<int:install_id>")
@login_required
def switch_install(install_id: int):
    inst = get_installation(install_id)
    if not inst or not inst.get("guest_url"):
        abort(404)

    target = _normalize_guest_url(inst["guest_url"])  # ← nettoyée à coup sûr

    origin = _current_origin()
    origin_label = get_app_setting("PRINTER_NAME","Mon installation")
    current_label = inst.get("label") or f"Installation #{install_id}"

    # ajoute uniquement nos 3 params “propres”
    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
    pr = urlparse(target)
    q = dict(parse_qsl(pr.query, keep_blank_values=True))
    q.update({
        "origin": origin,
        "origin_label": origin_label,
        "current_label": current_label,
    })
    target = urlunparse((pr.scheme, pr.netloc, pr.path, pr.params, urlencode(q), pr.fragment))
    return redirect(target, code=302)
