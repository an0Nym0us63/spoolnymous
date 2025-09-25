FROM python:3.14.0rc3-alpine3.22 AS python-builder

# Environnement
ENV APP_HOME=/home/app
ENV VIRTUAL_ENV=$APP_HOME/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install latest su-exec
RUN apk --no-cache add curl shadow; \
    set -ex; \
    curl -o /usr/local/bin/su-exec.c https://raw.githubusercontent.com/ncopa/su-exec/master/su-exec.c; \
    fetch_deps='gcc libc-dev'; \
    apk update && apk add --no-cache $fetch_deps; \
    rm -rf /var/lib/apt/lists/*; \
    gcc -Wall /usr/local/bin/su-exec.c -o/usr/local/bin/su-exec; \
    chown root:root /usr/local/bin/su-exec; \
    chmod 0755 /usr/local/bin/su-exec; \
    rm /usr/local/bin/su-exec.c;

# Add local user so we don't run as root
RUN groupmod -g 1000 users \
    && useradd -u 1000 -U app \
    && usermod -G users app \
    && mkdir -p $APP_HOME/static/prints \
    && mkdir -p $APP_HOME/logs \
    && mkdir -p /var/log/flask-app \
    && touch /var/log/flask-app/flask-app.err.log \
    && touch /var/log/flask-app/flask-app.out.log

WORKDIR $APP_HOME

# Dépendances système (ajout de ca-certificates et curl pour requêtes HTTPS fiables)
RUN apk update && apk add --no-cache ca-certificates \
    #libcurl4-openssl-dev \
    curl-dev \
    #libssl-dev \
    openssl-dev \
    ffmpeg 

# Dépendances Python
COPY --chown=app:app requirements.txt .
RUN python -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Code applicatif
COPY --chown=app:app . .

# Entrée
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]