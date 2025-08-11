# auth.py

from flask import Blueprint, render_template, request, redirect, url_for, flash, abort  # NEW: abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import secrets
from datetime import datetime, timedelta  # NEW

from config import DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, set_app_setting, get_all_app_settings

auth_bp = Blueprint('auth', __name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
GUEST_FILE = os.path.join(DATA_DIR, 'guest_tokens.json')  # NEW

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# =========================
# Modèle utilisateur
# =========================
class User(UserMixin):
    def __init__(self, username, role="user"):
        # Pour un invité, on peut stocker id="guest:<token>" pour éviter collision
        self.id = username
        self.role = role

    @property
    def is_guest(self):
        return self.role == "guest"

# =========================
# Utilitaires stockage
# =========================
def get_stored_user():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return None

def save_user(username, password, existing_data=None):
    password_hash = generate_password_hash(password)
    token = secrets.token_urlsafe(32)

    data = existing_data or get_stored_user() or {}
    data[username] = {
        "password_hash": password_hash,
        "token": token
    }
    with open(USERS_FILE, 'w') as f:
        json.dump(data, f)
    return token

def get_user_token(username):
    data = get_stored_user()
    if data and username in data:
        return data[username].get("token")
    return None

# ----- INVITÉS (guest) -----  # NEW
def _load_guest_tokens():
    if os.path.exists(GUEST_FILE):
        with open(GUEST_FILE, 'r') as f:
            return json.load(f)
    return {}

def _save_guest_tokens(tokens_dict):
    with open(GUEST_FILE, 'w') as f:
        json.dump(tokens_dict, f)

def create_guest_link(days_valid: int = 30) -> str:
    tokens = _load_guest_tokens()
    token = secrets.token_urlsafe(24)
    now = datetime.utcnow()
    expires_at = (now + timedelta(days=days_valid)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tokens[token] = {
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": expires_at,
        "revoked": False
    }
    _save_guest_tokens(tokens)
    return token

def list_guest_links():
    tokens = _load_guest_tokens()
    # retourne une liste exploitable en template
    out = []
    for tok, meta in tokens.items():
        out.append({
            "token": tok,
            "created_at": meta.get("created_at"),
            "expires_at": meta.get("expires_at"),
            "revoked": meta.get("revoked", False)
        })
    return out

def revoke_guest_link(token: str) -> bool:
    tokens = _load_guest_tokens()
    if token in tokens:
        tokens[token]["revoked"] = True
        _save_guest_tokens(tokens)
        return True
    return False

def _is_guest_token_valid(token: str) -> bool:
    tokens = _load_guest_tokens()
    meta = tokens.get(token)
    if not meta or meta.get("revoked"):
        return False
    exp = meta.get("expires_at")
    if exp:
        try:
            dt = datetime.strptime(exp, "%Y-%m-%dT%H:%M:%SZ")
            return datetime.utcnow() <= dt
        except Exception:
            return False
    return True

# =========================
# Auth
# =========================
def validate_credentials(username, password):
    user_data = get_stored_user()
    if user_data:
        stored_username = list(user_data.keys())[0]
        stored_hash = user_data[stored_username]['password_hash']
        if username == stored_username and check_password_hash(stored_hash, password):
            return True
    else:
        return username == DEFAULT_ADMIN_USERNAME and password == DEFAULT_ADMIN_PASSWORD
    return False

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if validate_credentials(username, password):
            user = User(username, role="user")
            login_user(user, remember=True)
            return redirect(url_for('home'))
        else:
            flash('Identifiants incorrects.', 'danger')
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

# =========================
# Lien invité (auto-login)
# =========================
@auth_bp.route("/guest/<token>")
def guest_autologin(token):
    if _is_guest_token_valid(token):
        # id distinct pour l’invité
        user = User(username=f"guest:{token}", role="guest")
        login_user(user, remember=False)
        # Reutilise ta page de redirection pour garder les thèmes/params
        return render_template("redirect_with_theme.html", query=request.query_string.decode())
    return "Lien invité invalide, expiré ou révoqué", 403

# =========================
# Settings (admin)
# =========================
@auth_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    # Interdire aux invités d’ouvrir la page settings
    if getattr(current_user, "is_guest", False):
        abort(403)

    user_data = get_stored_user()
    user_token = get_user_token(current_user.id if not current_user.is_guest else DEFAULT_ADMIN_USERNAME)

    if request.method == 'POST':
        # Régénération du token admin
        if request.form.get("regen_token") == "1":
            existing = get_stored_user()
            password_hash = existing[current_user.id]["password_hash"] if existing and current_user.id in existing else None
            if password_hash:
                token = secrets.token_urlsafe(32)
                existing[current_user.id]["token"] = token
                with open(USERS_FILE, 'w') as f:
                    json.dump(existing, f)
                flash("Token régénéré avec succès.", "success")
            return redirect(url_for('auth.settings'))

        # Mise à jour des paramètres système
        elif request.form.get("update_core_settings") == "1":
            for key in ["PRINTER_ID", "PRINTER_ACCESS_CODE", "PRINTER_IP", "SPOOLMAN_BASE_URL", "COST_BY_HOUR","LOCATION_MAPPING","AMS_ORDER","COST_BY_HOUR_PRINTER"]:
                value = request.form.get(key, "").strip()
                set_app_setting(key, value)
            flash("Paramètres système mis à jour avec succès ✅", "success")
            return redirect(url_for('auth.settings'))

        # Changement de mot de passe
        elif request.form.get("new_password") and request.form.get("confirm_password"):
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')
            if new_password != confirm_password:
                flash("Les mots de passe ne correspondent pas.", "warning")
            else:
                save_user(current_user.id, new_password)
                flash("Mot de passe mis à jour.", "success")
                return redirect(url_for('auth.settings'))

        # NEW: Créer lien invité
        elif request.form.get("create_guest_link") == "1":
            days = int(request.form.get("guest_days_valid", "30") or "30")
            tok = create_guest_link(days_valid=days)
            flash("Lien invité créé ✅", "success")
            return redirect(url_for('auth.settings'))

        # NEW: Révoquer un lien invité
        elif request.form.get("revoke_guest_token"):
            tok = request.form.get("revoke_guest_token")
            if revoke_guest_link(tok):
                flash("Lien invité révoqué ✅", "success")
            else:
                flash("Échec de révocation (token inconnu).", "warning")
            return redirect(url_for('auth.settings'))

    return render_template(
        "settings.html",
        user=current_user,
        using_default=(user_data is None),
        token=user_token,
        settings=get_all_app_settings(),
        page_title="Paramètres",
        guest_links=list_guest_links(),  # NEW: pour afficher dans le template
    )

@auth_bp.route("/autologin/<token>")
def autologin_token(token):
    users = get_stored_user()
    if users:
        for username, info in users.items():
            if info.get("token") == token:
                user = User(username, role="user")
                login_user(user, remember=True)
                return render_template("redirect_with_theme.html", query=request.query_string.decode())
    return "Token invalide ou expiré", 403

# =========================
# Sécurité lecture seule
# =========================
# Blocage GLOBAL des méthodes d’écriture pour les invités
@auth_bp.before_app_request
def block_guest_writes():
    if current_user.is_authenticated and getattr(current_user, "is_guest", False):
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            # Autoriser POST uniquement sur /auth/settings pour créer/révoquer ? Non => invités n’y ont pas accès de toute façon.
            abort(403)

# Décorateur optionnel si certaines routes GET déclenchent des actions
def write_required(view_func):
    from functools import wraps
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if current_user.is_authenticated and getattr(current_user, "is_guest", False):
            abort(403)
        return view_func(*args, **kwargs)
    return wrapper
