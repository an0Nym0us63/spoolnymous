# remote.py
from flask import Blueprint, render_template, abort
from flask_login import login_required, current_user
from installations import get_installation

remote_bp = Blueprint('remote', __name__)

@remote_bp.route('/remote/<int:install_id>')
@login_required
def remote_view(install_id: int):
    # Optionnel : autoriser les invités à consulter d’autres installs
    # Ici on autorise tous les users loggés.
    inst = get_installation(install_id)
    if not inst or not inst.get("guest_url"):
        abort(404)
    # L’URL est un lien invité vers l’installation distante (autologin invité)
    return render_template('remote_view.html',
                           remote_label=inst.get("label") or f"Installation #{install_id}",
                           remote_url=inst["guest_url"])
