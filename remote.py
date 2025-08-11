# remote.py
from flask import Blueprint, render_template, abort
from flask_login import login_required, current_user
from installations import get_installation
from urllib.parse import urlsplit, urlunsplit

remote_bp = Blueprint('remote', __name__)

def _force_https(u: str) -> str:
    try:
        p = urlsplit(u)
        if p.scheme == 'http':
            p = p._replace(scheme='https')
            return urlunsplit(p)
    except Exception:
        pass
    return u

@remote_bp.route('/remote/<int:install_id>')
@login_required
def remote_view(install_id: int):
    inst = get_installation(install_id)
    if not inst or not inst.get("guest_url"):
        abort(404)
    remote_url = _force_https(inst["guest_url"])
    return render_template('remote_view.html',
                           remote_label=inst.get("label") or f"Installation #{install_id}",
                           remote_url=remote_url)
