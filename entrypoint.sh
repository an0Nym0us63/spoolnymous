#!/bin/sh

# UID/GID par défaut
PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Mise à jour de l'utilisateur app si existant
if getent group app >/dev/null 2>&1; then
    groupmod -o -g "$PGID" app
fi

if id app >/dev/null 2>&1; then
    usermod -o -u "$PUID" -g "$PGID" app
fi

echo "[ENTRYPOINT] UID=$(id -u app), GID=$(id -g app)"
echo "[ENTRYPOINT] Lancement de Gunicorn..."
chown -R app:app /home/app/data /home/app/logs /home/app/static/prints
# Lancement de Gunicorn avec le bon user et module app.py
exec su-exec app gunicorn -k gthread -w 1 --threads 8 --timeout 0 \
  --max-requests 2000 --max-requests-jitter 200 \
  -b 0.0.0.0:8000 app:app "$@"