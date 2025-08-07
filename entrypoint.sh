#!/bin/sh

# UID/GID par défaut si non fournis
PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Met à jour l'UID/GID de l'utilisateur app
if getent group app >/dev/null 2>&1; then
    groupmod -o -g "$PGID" app
fi

if id app >/dev/null 2>&1; then
    usermod -o -u "$PUID" -g "$PGID" app
fi

echo "[ENTRYPOINT] UID=$(id -u app), GID=$(id -g app)"
echo "[ENTRYPOINT] Lancement de Gunicorn..."

exec su-exec app gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 src.app:app "$@"
