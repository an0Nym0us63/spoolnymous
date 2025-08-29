#!/bin/sh
set -eu

# --------- Config repo en dur ---------
GH_OWNER="an0Nym0us63"
GH_REPO="spoolnymous"

# --------- Branche à suivre (venant de la stack) ---------
BUILD_BRANCH="${BUILD_BRANCH:-release}"

# Désactiver complètement les checks réseau si demandé
DISABLE_UPDATE_CHECK="${DISABLE_UPDATE_CHECK:-0}"

# --------- UID/GID comme avant ---------
PUID=${PUID:-1000}
PGID=${PGID:-1000}

if getent group app >/dev/null 2>&1; then
    groupmod -o -g "$PGID" app || true
fi

if id app >/dev/null 2>&1; then
    usermod -o -u "$PUID" -g "$PGID" app || true
fi

# --------- Utilitaires ---------
http_get() {
    # $1 = URL
    if command -v curl >/dev/null 2>&1; then
        # -fsSL silencieux, --max-time 2 pour ne pas bloquer le boot
        curl -fsSL --max-time 2 "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -T 2 -O - "$1"
    else
        return 127
    fi
}

extract_sha() {
    # Extrait le premier SHA (40 hex) du JSON GitHub sur stdin
    sed -n 's/.*"sha":[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' | head -n 1
}

# --------- Commit / Date (écrits pour l'app) ---------
COMMIT_FILE="/etc/image_commit_sha"
DATE_FILE="/etc/image_build_date"

# Toujours initialiser rapidement (non bloquant)
printf "%s" "${COMMIT_SHA:-unknown}" | cut -c1-7 > "$COMMIT_FILE" || true
if [ -n "${BUILD_DATE:-}" ]; then
  printf "%s" "$BUILD_DATE" > "$DATE_FILE" || true
else
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$DATE_FILE" || true
fi

# Mise à jour en arrière-plan si autorisée et si COMMIT_SHA non fourni
if [ "${DISABLE_UPDATE_CHECK}" != "1" ] && [ -z "${COMMIT_SHA:-}" ]; then
  (
    API_URL="https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/commits/${BUILD_BRANCH}"
    if JSON="$(http_get "$API_URL" || true)"; then
      FULL_SHA="$(printf "%s" "$JSON" | extract_sha || true)"
      if [ -n "${FULL_SHA:-}" ]; then
        printf "%s" "${FULL_SHA}" | cut -c1-7 > "$COMMIT_FILE" || true
      fi
    fi
  ) >/dev/null 2>&1 &
fi

# --------- Logs & perms ---------
echo "[ENTRYPOINT] BRANCH=${BUILD_BRANCH}  COMMIT=$(cat "$COMMIT_FILE" || echo 'unknown')  DATE=$(cat "$DATE_FILE" || echo 'unknown')"
echo "[ENTRYPOINT] UID=$(id -u app 2>/dev/null || echo '?'), GID=$(id -g app 2>/dev/null || echo '?')"

chown -R app:app \
  /home/app/data \
  /home/app/logs \
  /home/app/static/prints \
  /home/app/static/uploads 2>/dev/null || true

# --------- Lancement Gunicorn ---------
echo "[ENTRYPOINT] Lancement de Gunicorn..."
# 2 workers pour éviter qu'une requête lente bloque l'UI
exec su-exec app gunicorn -k gthread -w 1 --threads 10 --timeout 120 \
  -b 0.0.0.0:8000 app:app "$@"
