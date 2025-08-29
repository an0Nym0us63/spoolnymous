#!/bin/sh
set -eu

# --------- Config repo en dur ---------
GH_OWNER="an0Nym0us63"
GH_REPO="spoolnymous"

# --------- Branche à suivre (venant de la stack) ---------
BUILD_BRANCH="${BUILD_BRANCH:-release}"

# --------- UID/GID comme avant ---------
PUID=${PUID:-1000}
PGID=${PGID:-1000}

if getent group app >/dev/null 2>&1; then
    groupmod -o -g "$PGID" app
fi

if id app >/dev/null 2>&1; then
    usermod -o -u "$PUID" -g "$PGID" app
fi

# --------- Utilitaires ---------
http_get() {
    # $1 = URL
    if command -v curl >/dev/null 2>&1; then
        # -fsSL silencieux, --max-time 5 pour ne pas bloquer le boot
        curl -fsSL --max-time 5 "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -T 5 -O - "$1"
    else
        return 127
    fi
}

extract_sha() {
    # Extrait le premier SHA (40 hex) du JSON GitHub sur stdin
    # On prend le premier "sha" rencontré (HEAD du commit demandé)
    sed -n 's/.*"sha":[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' | head -n 1
}

# --------- Commit / Date (écrits pour l'app) ---------
COMMIT_FILE="/etc/image_commit_sha"
DATE_FILE="/etc/image_build_date"

SHORT_SHA=""

# 1) Si COMMIT_SHA fourni via env, on le respecte
if [ -n "${COMMIT_SHA:-}" ]; then
    SHORT_SHA=$(printf "%s" "$COMMIT_SHA" | cut -c1-7)
else
    # 2) Sinon on tente GitHub (non bloquant)
    API_URL="https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/commits/${BUILD_BRANCH}"
    if JSON="$(http_get "$API_URL" || true)"; then
        FULL_SHA="$(printf "%s" "$JSON" | extract_sha || true)"
        if [ -n "${FULL_SHA:-}" ]; then
            SHORT_SHA=$(printf "%s" "$FULL_SHA" | cut -c1-7)
        fi
    fi
fi

if [ -n "${SHORT_SHA:-}" ]; then
    printf "%s" "$SHORT_SHA" > "$COMMIT_FILE"
else
    printf "unknown" > "$COMMIT_FILE"
fi

# Date de build : env prioritaire sinon horodatage actuel UTC
if [ -n "${BUILD_DATE:-}" ]; then
    printf "%s" "$BUILD_DATE" > "$DATE_FILE"
else
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$DATE_FILE"
fi

# --------- Logs & perms ---------
echo "[ENTRYPOINT] BRANCH=${BUILD_BRANCH}  COMMIT=$(cat "$COMMIT_FILE")  DATE=$(cat "$DATE_FILE")"
echo "[ENTRYPOINT] UID=$(id -u app), GID=$(id -g app)"

chown -R app:app \
  /home/app/data \
  /home/app/logs \
  /home/app/static/prints \
  /home/app/static/uploads || true

# --------- Lancement Gunicorn (inchangé) ---------
echo "[ENTRYPOINT] Lancement de Gunicorn..."
exec su-exec app gunicorn -k gthread -w 1 --threads 10 --timeout 120 \
  -b 0.0.0.0:8000 app:app "$@"
