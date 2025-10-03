#!/bin/sh
set -eu

# --------- Config repo ---------
GH_OWNER="an0Nym0us63"
GH_REPO="spoolnymous"
BUILD_BRANCH="${BUILD_BRANCH:-release}"
DISABLE_UPDATE_CHECK="${DISABLE_UPDATE_CHECK:-0}"

# --------- UID/GID ---------
PUID=${PUID:-1000}
PGID=${PGID:-1000}

# --------- Gunicorn tunables ---------
GWORKERS="${GWORKERS:-1}"          # 1 recommandé (MQTT-safe)
GTHREADS="${GTHREADS:-10}"
GTIMEOUT="${GTIMEOUT:-120}"
GBIND="${GBIND:-0.0.0.0:8000}"
GEXTRA_OPTS="${GEXTRA_OPTS:-}"     # ex: "--access-logfile - --error-logfile -"

# --------- Utils ---------
http_get() {
  # $1 URL
  if command -v curl >/dev/null 2>&1; then
    # token GitHub optionnel pour éviter le rate limit
    if [ -n "${GITHUB_TOKEN:-}" ]; then
      curl -fsSL --connect-timeout 2 --max-time 4 \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        "$1"
    else
      curl -fsSL --connect-timeout 2 --max-time 4 \
        -H "Accept: application/vnd.github+json" \
        "$1"
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

# --------- Commit / Date ---------
COMMIT_FILE="/etc/image_commit_sha"
DATE_FILE="/etc/image_build_date"
printf "%s" "${COMMIT_SHA:-unknown}" | cut -c1-7 > "$COMMIT_FILE" || true
if [ -n "${BUILD_DATE:-}" ]; then
  printf "%s" "$BUILD_DATE" > "$DATE_FILE" || true
else
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$DATE_FILE" || true
fi

if [ "$DISABLE_UPDATE_CHECK" != "1" ] && [ -z "${COMMIT_SHA:-}" ]; then
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

# --------- Affichage env ---------
echo "[ENTRYPOINT] BRANCH=${BUILD_BRANCH}  COMMIT=$(cat "$COMMIT_FILE" || echo 'unknown')  DATE=$(cat "$DATE_FILE" || echo 'unknown')"

# --------- Prépare user/group si présents ---------
if command -v getent >/dev/null 2>&1 && getent group app >/dev/null 2>&1; then
  groupmod -o -g "$PGID" app || true
fi
if command -v id >/dev/null 2>&1 && id app >/dev/null 2>&1; then
  usermod -o -u "$PUID" -g "$PGID" app || true
fi

# --------- Dossiers & perms ---------
mkdir -p /home/app/data /home/app/logs /home/app/static/prints /home/app/static/uploads /var/log/flask-app
: > /var/log/flask-app/flask-app.err.log || true
: > /var/log/flask-app/flask-app.out.log || true

echo "[ENTRYPOINT] UID(app)=$(id -u app 2>/dev/null || echo '?'), GID(app)=$(id -g app 2>/dev/null || echo '?')"
chown -R app:app \
  /home/app/data \
  /home/app/logs \
  /home/app/static/prints \
  /home/app/static/uploads \
  /var/log/flask-app 2>/dev/null || true

# --------- Drop privileges helper ---------
run_as_app() {
  if [ "$(id -u)" -ne 0 ]; then
    # Déjà non-root
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

# --------- Lancement Gunicorn ---------
echo "[ENTRYPOINT] Lancement Gunicorn: workers=${GWORKERS} threads=${GTHREADS} timeout=${GTIMEOUT} bind=${GBIND}"
export PYTHONUNBUFFERED=1
run_as_app gunicorn -k gthread -w "$GWORKERS" --threads "$GTHREADS" \
  --timeout "$GTIMEOUT" -b "$GBIND" app:app $GEXTRA_OPTS "$@"
