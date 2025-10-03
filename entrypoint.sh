#!/bin/sh
set -eu

GH_OWNER="an0Nym0us63"
GH_REPO="spoolnymous"
BUILD_BRANCH="${BUILD_BRANCH:-release}"
DISABLE_UPDATE_CHECK="${DISABLE_UPDATE_CHECK:-0}"

# Gunicorn
GWORKERS="${GWORKERS:-1}"
GTHREADS="${GTHREADS:-10}"
GTIMEOUT="${GTIMEOUT:-120}"
GBIND="${GBIND:-0.0.0.0:8000}"
GEXTRA_OPTS="${GEXTRA_OPTS:-}"

APP_HOME="/home/app"
PUID=${PUID:-1000}
PGID=${PGID:-1000}
UMASK="${UMASK:-0002}"

IS_ROOT=0
[ "$(id -u)" -eq 0 ] && IS_ROOT=1
umask "$UMASK"

# --- Commit/date files (root -> /etc, sinon -> $APP_HOME/run) ---
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

# --- user/group : s'assure que app:app existe et a les bons uid/gid ---
ensure_user_group() {
  if [ "$IS_ROOT" -eq 1 ]; then
    # groupe
    if getent group app >/dev/null 2>&1; then
      groupmod -o -g "$PGID" app || true
    else
      addgroup -g "$PGID" app || true
    fi
    # user
    if id app >/dev/null 2>&1; then
      usermod -o -u "$PUID" -g "$PGID" app || true
    else
      adduser -D -H -s /sbin/nologin -G app -u "$PUID" app || true
    fi
  else
    # Non-root : on ne peut pas créer/modifier, on espère que l'image l’a déjà
    :
  fi
}

ensure_dir() {
  # $1=dir  $2=owner  $3=group  $4=mode
  if [ "$IS_ROOT" -eq 1 ]; then
    install -d -o "${2:-app}" -g "${3:-app}" -m "${4:-0775}" "$1"
  else
    mkdir -p "$1"
    chmod "${4:-0775}" "$1" 2>/dev/null || true
  fi
}

fix_perms_tree() {
  # $@=dirs
  [ "$IS_ROOT" -eq 1 ] || return 0
  for d in "$@"; do
    [ -d "$d" ] || continue
    chown -R app:app "$d" 2>/dev/null || true
    find "$d" -type d -exec chmod 0775 {} + 2>/dev/null || true
    find "$d" -type f -exec chmod 0664 {} + 2>/dev/null || true
  done
}

check_writable() {
  # log si non-writable (non-root)
  [ "$IS_ROOT" -eq 1 ] && return 0
  local err=0
  for p in "$@"; do
    if ! [ -w "$p" ]; then
      err=1
      echo "[ENTRYPOINT][WARN] Non inscriptible: $p ($(ls -ld "$p" 2>/dev/null || echo '?'))" >&2
    fi
  done
  [ $err -eq 0 ]
}

# --- init commit/date (non bloquant) ---
printf "%s" "${COMMIT_SHA:-unknown}" | cut -c1-7 > "$COMMIT_FILE" 2>/dev/null || true
if [ -n "${BUILD_DATE:-}" ]; then
  printf "%s" "$BUILD_DATE" > "$DATE_FILE" 2>/dev/null || true
else
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$DATE_FILE" 2>/dev/null || true
fi
if [ "$DISABLE_UPDATE_CHECK" != "1" ] && [ -z "${COMMIT_SHA:-}" ]; then
  API_URL="https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/commits/${BUILD_BRANCH}"
  if JSON="$(http_get "$API_URL" || true)"; then
    FULL_SHA="$(printf "%s" "$JSON" | extract_sha || true)"
    [ -n "${FULL_SHA:-}" ] && printf "%s" "${FULL_SHA}" | cut -c1-7 > "$COMMIT_FILE" 2>/dev/null || true
  fi
fi

# --- user/group + dossiers ---
ensure_user_group

ensure_dir "$APP_HOME/data"            app app 0775
ensure_dir "$APP_HOME/logs"            app app 0775
ensure_dir "$APP_HOME/static/prints"   app app 0775
ensure_dir "$APP_HOME/static/uploads"  app app 0775
ensure_dir /var/log/flask-app          root root 0775
: > /var/log/flask-app/flask-app.err.log 2>/dev/null || true
: > /var/log/flask-app/flask-app.out.log 2>/dev/null || true

fix_perms_tree "$APP_HOME/data" "$APP_HOME/logs" "$APP_HOME/static/prints" "$APP_HOME/static/uploads"
check_writable "$APP_HOME/data" "$APP_HOME/logs" "$APP_HOME/static/prints" "$APP_HOME/static/uploads" || {
  echo "[ENTRYPOINT][HINT] Si tu utilises user: \"1000:1000\" dans Compose, corrige côté hôte :" >&2
  echo "  chown -R 1000:1000 ./data ./static/uploads ./static/prints" >&2
}

echo "[ENTRYPOINT] BRANCH=${BUILD_BRANCH}  COMMIT=$(cat "$COMMIT_FILE" 2>/dev/null || echo 'unknown')  DATE=$(cat "$DATE_FILE" 2>/dev/null || echo 'unknown')"
echo "[ENTRYPOINT] UID(app)=$(id -u app 2>/dev/null || echo '?'), GID(app)=$(id -g app 2>/dev/null || echo '?')"

# --- drop helper : exige su-exec/gosu en root ---
drop_and_exec() {
  if [ "$IS_ROOT" -eq 0 ]; then
    exec "$@"
  fi
  if command -v su-exec >/dev/null 2>&1; then
    exec su-exec app:app "$@"
  elif command -v gosu >/dev/null 2>&1; then
    exec gosu app:app "$@"
  else
    echo "[ENTRYPOINT][FATAL] Impossible de drop les privilèges (ni su-exec ni gosu). Abandon." >&2
    exit 90
  fi
}

# --- Gunicorn (toujours en app:app) ---
echo "[ENTRYPOINT] Lancement Gunicorn: workers=${GWORKERS} threads=${GTHREADS} timeout=${GTIMEOUT} bind=${GBIND}"
export PYTHONUNBUFFERED=1
export HOME="$APP_HOME"
drop_and_exec gunicorn -k gthread \
  -w "$GWORKERS" --threads "$GTHREADS" --timeout "$GTIMEOUT" \
  -b "$GBIND" --user app --group app \
  app:app $GEXTRA_OPTS "$@"
