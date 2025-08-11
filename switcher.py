# switcher.py
from flask import Blueprint, redirect, abort, request, current_app
from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qsl, urlparse, urlunparse
from flask_login import login_required
from installations import get_installation
from config import get_app_setting  # si tu as déjà ça

switch_bp = Blueprint('switch', __name__)

def _force_https(u: str) -> str:
    try:
        p = urlsplit(u.strip())
        if p.scheme == 'http':
            p = p._replace(scheme='https')
            return urlunsplit(p)
    except Exception:
        pass
    return u

def _append_params(url: str, extra: dict) -> str:
    """Ajoute/merge des query params proprement."""
    pr = urlparse(url)
    q = dict(parse_qsl(pr.query, keep_blank_values=True))
    q.update({k: v for k, v in extra.items() if v is not None and v != ""})
    new_qs = urlencode(q, doseq=True)
    return urlunparse((pr.scheme, pr.netloc, pr.path, pr.params, new_qs, pr.fragment))

def _current_origin() -> str:
    # BASE_URL si configuré, sinon on reconstruit depuis la requête
    base = get_app_setting("BASE_URL", "").strip() if 'get_app_setting' in globals() else ""
    if base:
        return base
    scheme = "https" if request.headers.get("X-Forwarded-Proto","").lower()=="https" or request.is_secure else "http"
    return f"{scheme}://{request.host}/"

@switch_bp.route("/switch/<int:install_id>")
@login_required
def switch_install(install_id: int):
    inst = get_installation(install_id)
    if not inst or not inst.get("guest_url"):
        abort(404)
    target = _force_https(inst["guest_url"])

    # Paramètres à propager
    origin = _current_origin()
    origin_label = get_app_setting("PRINTER_NAME", "Mon installation")
    current_label = inst.get("label") or f"Installation #{install_id}"  # 👈 nom tel que défini CHEZ TOI

    target = _append_params(target, {
        "origin": origin,
        "origin_label": origin_label,
        "current_label": current_label,  # 👈 on envoie aussi le label cible
    })
    return redirect(target, code=302)

