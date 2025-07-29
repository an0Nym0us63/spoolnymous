# auth.py

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import secrets

from config import DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD

auth_bp = Blueprint('auth', __name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')

# Assurer que le dossier data existe
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class User(UserMixin):
    def __init__(self, username):
        self.id = username

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
            user = User(username)
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

@auth_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user_data = get_stored_user()

    if request.method == 'POST':
        if request.form.get("regen_token") == "1":
            existing = get_stored_user()
            password_hash = existing[current_user.id]["password_hash"] if existing and current_user.id in existing else None
            if password_hash:
                # Regénérer uniquement le token, garder le même mot de passe
                token = secrets.token_urlsafe(32)
                existing[current_user.id]["token"] = token
                with open(USERS_FILE, 'w') as f:
                    json.dump(existing, f)
                flash("Token régénéré avec succès.", "success")
        else:
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')
            if new_password != confirm_password:
                flash("Les mots de passe ne correspondent pas.", "warning")
            else:
                save_user(current_user.id, new_password)
                flash("Mot de passe mis à jour.", "success")
                return redirect(url_for('auth.settings'))

    user_token = get_user_token(current_user.id)
    return render_template("settings.html", user=current_user, using_default=(user_data is None), token=user_token)


@auth_bp.route("/autologin/<token>")
def autologin_token(token):
    users = get_stored_user()
    if users:
        for username, info in users.items():
            if info.get("token") == token:
                user = User(username)
                login_user(user, remember=True)
                return redirect(url_for("home"))
    return "Token invalide ou expiré", 403
