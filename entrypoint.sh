#!/bin/sh
set -eu

# --- Repo / branche ---
GH_OWNER="an0Nym0us63"
GH_REPO="spoolnymous"
BUILD_BRANCH="${BUILD_BRANCH:-release}"
DISABLE_UPDATE_CHECK="${DISABLE_UPDATE_CHECK:-0}"

# --- UID/GID comme avant ---
PUID=${PUID:-1000}
PGID=${PGID:-1000}
UMASK="${UMASK:-0002}"

APP_HOME="/home/app"
IS_ROOT=0
[ "$(id -u)" -eq 0 ] && IS_ROOT=1
umask "$UMASK"

# --- commit/date files (root => /etc, sinon => $APP_HOME/run) ---
if [ "$IS_ROOT" -eq 1 ]; then
  COMMIT_FILE="/etc/image_commit_sha"
  DATE_FILE="/etc/image_build_date"
else
  mkdir -p "$APP_HOME/run"
  COMMIT_FILE="$APP_HOME/run/image_commit_sha"
  DATE_FILE="$APP_HOME/run/image_build_date"
fi

# --- utils ---
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
extract_sha(){ sed -n 's/.*"sha":[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' | head -n 1; }

# --- user/group (Debian) ---
if [ "$IS_ROOT" -eq 1 ]; then
  if getent group app >/dev/null 2>&1; then
    groupmod -o -g "$PGID" app || true
  else
    groupadd -g "$PGID" app || true
  fi
  if id app >/dev/null 2>&1; then
    usermod -o -u "$PUID" -g "$PGID" app || true
  else
    useradd -u "$PUID" -g "$PGID" -M -d "$APP_HOME" -s /usr/sbin/nologin app || true
  fi
fi

# --- commit/date init (synchrone, rapide) ---
SHORT_SHA="${COMMIT_SHA:-}"
if [ -z "$SHORT_SHA" ] && [ "$DISABLE_UPDATE_CHECK" != "1" ]; then
  API_URL="https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/commits/${BUILD_BRANCH}"
  JSON="$(http_get "$API_URL" || true)"
  FULL_SHA="$(printf "%s" "$JSON" | extract_sha || true)"
  [ -n "$FULL_SHA" ] && SHORT_SHA="$(printf "%s" "$FULL_SHA" | cut -c1-7)"
fi
[ -n "$SHORT_SHA" ] || SHORT_SHA="unknown"
printf "%s" "$SHORT_SHA" > "$COMMIT_FILE" 2>/dev/null || true

if [ -n "${BUILD_DATE:-}" ]; then
  printf "%s" "$BUILD_DATE" > "$DATE_FILE" 2>/dev/null || true
else
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$DATE_FILE" 2>/dev/null || true
fi

# IMPORTANT : ton app lit l’ENV directement → on exporte la VALEUR (pas un chemin)
export IMAGE_COMMIT_FILE="$SHORT_SHA"
export IMAGE_BUILD_DATE_FILE="$(cat "$DATE_FILE" 2>/dev/null || echo unknown)"

echo "[ENTRYPOINT] BRANCH=${BUILD_BRANCH}  COMMIT=${SHORT_SHA}  DATE=$(cat "$DATE_FILE" 2>/dev/null || echo 'unknown')"

# --- dossiers + permissions (répare aussi les volumes montés) ---
make_dir() { # dir owner group mode (2775 sur dossiers -> SGID)
  if [ "$IS_ROOT" -eq 1 ]; then
    install -d -o "${2:-app}" -g "${3:-app}" -m "${4:-2775}" "$1"
  else
    mkdir -p "$1"; chmod "${4:-2775}" "$1" 2>/dev/null || true
  fi
}

make_dir "$APP_HOME/data"            app app 2775
make_dir "$APP_HOME/logs"            app app 2775
make_dir "$APP_HOME/static/prints"   app app 2775
make_dir "$APP_HOME/static/uploads"  app app 2775
make_dir /var/log/flask-app          root root 2775
: > /var/log/flask-app/flask-app.err.log 2>/dev/null || true
: > /var/log/flask-app/flask-app.out.log 2>/dev/null || true

if [ "$IS_ROOT" -eq 1 ]; then
  chown -R app:app \
    "$APP_HOME/data" \
    "$APP_HOME/logs" \
    "$APP_HOME/static/prints" \
    "$APP_HOME/static/uploads" 2>/dev/null || true

  # SGID sur dossiers, 0664 sur fichiers
  find "$APP_HOME/data" "$APP_HOME/logs" "$APP_HOME/static/prints" "$APP_HOME/static/uploads" \
    -type d -exec chmod 2775 {} + 2>/dev/null || true
  find "$APP_HOME/data" "$APP_HOME/logs" "$APP_HOME/static/prints" "$APP_HOME/static/uploads" \
    -type f -exec chmod 0664 {} + 2>/dev/null || true
fi

echo "[ENTRYPOINT] UID(app)=$(id -u app 2>/dev/null || echo '?'), GID(app)=$(id -g app 2>/dev/null || echo '?')"

# --- Gunicorn (1 worker MQTT-safe, comme avant) ---
echo "[ENTRYPOINT] Lancement de Gunicorn (1 worker)..."
exec \
  ${IS_ROOT:+gosu app:app} \
  gunicorn -k gthread -w 1 --threads 10 --timeout 120 \
  -b 0.0.0.0:8000 app:app "$@"
