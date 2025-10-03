#!/bin/sh
set -eu

GH_OWNER="an0Nym0us63"
GH_REPO="spoolnymous"
BUILD_BRANCH="${BUILD_BRANCH:-release}"
DISABLE_UPDATE_CHECK="${DISABLE_UPDATE_CHECK:-0}"

PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Gunicorn
GWORKERS="${GWORKERS:-1}"
GTHREADS="${GTHREADS:-10}"
GTIMEOUT="${GTIMEOUT:-120}"
GBIND="${GBIND:-0.0.0.0:8000}"
GEXTRA_OPTS="${GEXTRA_OPTS:-}"

APP_HOME="/home/app"
IS_ROOT=0
[ "$(id -u)" -eq 0 ] && IS_ROOT=1

# --- Choix des fichiers "commit/date" en fonction des droits ---
# Si root: on garde /etc/*. Sinon: fallback dans $APP_HOME/run/
COMMIT_FILE_DEFAULT="/etc/image_commit_sha"
DATE_FILE_DEFAULT="/etc/image_build_date"
if [ "$IS_ROOT" -eq 1 ]; then
  COMMIT_FILE="${IMAGE_COMMIT_FILE:-$COMMIT_FILE_DEFAULT}"
  DATE_FILE="${IMAGE_BUILD_DATE_FILE:-$DATE_FILE_DEFAULT}"
else
  mkdir -p "$APP_HOME/run"
  COMMIT_FILE="${IMAGE_COMMIT_FILE:-$APP_HOME/run/image_commit_sha}"
  DATE_FILE="${IMAGE_BUILD_DATE_FILE:-$APP_HOME/run/image_build_date}"
fi
export IMAGE_COMMIT_FILE="$COMMIT_FILE"
export IMAGE_BUILD_DATE_FILE="$DATE_FILE"

# --- helpers ---
http_get() {
  if command -v curl >/dev/null 2>&1; then
    if [ -n "${GITHUB_TOKEN:-}" ]; then
      curl -fsSL --connect-timeout 2 --max-time 4 \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" "$1"
    else
      curl -fsSL --connect-timeout 2 --max-time 4 \
        -H "Accept: application/vnd.github+json" "$1"
    fi
  elif command -v wget >/dev/null 2>&1; then
    wget -q -T 4 -O - "$1"
  else
    return 127
  fi
}
extract_sha() {
  sed -n 's/.*"sha":[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' | head -n 1
}

# --- Commit/Date (toujours non bloquant) ---
printf "%s" "${COMMIT_SHA:-unknown}" | cut -c1-7 > "$COMMIT_FILE" 2>/dev/null || true
if [ -n "${BUILD_DATE:-}" ]; then
  printf "%s" "$BUILD_DATE" > "$DATE_FILE" 2>/dev/null || true
else
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$DATE_FILE" 2>/dev/null || true
fi

if [ "$DISABLE_UPDATE_CHECK" != "1" ] && [ -z "${COMMIT_SHA:-}" ]; then
  (
    API_URL="https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/commits/${BUILD_BRANCH}"
    if JSON="$(http_get "$API_URL" || true)"; then
      FULL_SHA="$(printf "%s" "$JSON" | extract_sha || true)"
      if [ -n "${FULL_SHA:-}" ]; then
        printf "%s" "${FULL_SHA}" | cut -c1-7 > "$COMMIT_FILE" 2>/dev/null || true
      fi
    fi
  ) >/dev/null 2>&1 &
fi

# --- Logs env ---
echo "[ENTRYPOINT] BRANCH=${BUILD_BRANCH}  COMMIT=$(cat "$COMMIT_FILE" 2>/dev/null || echo 'unknown')  DATE=$(cat "$DATE_FILE" 2>/dev/null || echo 'unknown')"

# --- UID/GID & perms uniquement si root ---
if [ "$IS_ROOT" -eq 1 ]; then
  if command -v getent >/dev/null 2>&1 && getent group app >/dev/null 2>&1; then
    groupmod -o -g "$PGID" app || true
  fi
  if command -v id >/dev/null 2>&1 && id app >/dev/null 2>&1; then
    usermod -o -u "$PUID" -g "$PGID" app || true
  fi
fi

# --- Dossiers & ownership (chown seulement si root) ---
mkdir -p "$APP_HOME/data" "$APP_HOME/logs" "$APP_HOME/static/prints" "$APP_HOME/static/uploads" /var/log/flask-app
: > /var/log/flask-app/flask-app.err.log || true
: > /var/log/flask-app/flask-app.out.log || true

if [ "$IS_ROOT" -eq 1 ]; then
  chown -R app:app \
    "$APP_HOME/data" \
    "$APP_HOME/logs" \
    "$APP_HOME/static/prints" \
    "$APP_HOME/static/uploads" \
    /var/log/flask-app 2>/dev/null || true
fi

echo "[ENTRYPOINT] UID(app)=$(id -u app 2>/dev/null || echo '?'), GID(app)=$(id -g app 2>/dev/null || echo '?')"

# --- drop privileges helper ---
run_as_app() {
  if [ "$IS_ROOT" -eq 0 ]; then
    exec "$@"
  fi
  if command -v su-exec >/dev/null 2>&1; then
    exec su-exec app "$@"
  elif command -v gosu >/dev/null 2>&1; then
    exec gosu app "$@"
  else
    echo "[ENTRYPOINT][WARN] ni su-exec ni gosu trouvés, lancement en root" >&2
    exec "$@"
  fi
}

# --- Lancement Gunicorn ---
echo "[ENTRYPOINT] Lancement Gunicorn: workers=${GWORKERS} threads=${GTHREADS} timeout=${GTIMEOUT} bind=${GBIND}"
export PYTHONUNBUFFERED=1
run_as_app gunicorn -k gthread -w "$GWORKERS" --threads "$GTHREADS" \
  --timeout "$GTIMEOUT" -b "$GBIND" app:app $GEXTRA_OPTS "$@"
