#!/bin/sh

PUID=${PUID:-0}
PGID=${PGID:-0}

# Crée ou modifie le user `app` dynamiquement si nécessaire
if ! id "app" >/dev/null 2>&1; then
  addgroup -g "$PGID" app
  adduser -D -u "$PUID" -G app app
else
  groupmod -o -g "$PGID" app
  usermod -o -u "$PUID" app
fi

echo "[ENTRYPOINT] UID=$(id -u app), GID=$(id -g app)"
echo "[ENTRYPOINT] Démarrage de Gunicorn..."

exec su-exec app gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 src.app:app "$@"
