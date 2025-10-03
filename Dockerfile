FROM python:3.14.0rc3-alpine3.22 AS python-builder

# Environnement
ENV APP_HOME=/home/app
ENV VIRTUAL_ENV=$APP_HOME/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# su-exec est packagé sur Alpine → pas besoin de compiler
RUN apk add --no-cache curl shadow su-exec

# Add local user so we don't run as root
RUN groupmod -g 1000 users \
    && useradd -u 1000 -U app \
    && usermod -G users app \
    && mkdir -p $APP_HOME/static/prints $APP_HOME/static/uploads \
    && mkdir -p $APP_HOME/logs \
    && mkdir -p /var/log/flask-app \
    && touch /var/log/flask-app/flask-app.err.log \
    && touch /var/log/flask-app/flask-app.out.log

WORKDIR $APP_HOME

# Dépendances système (ffmpeg + libheif pour HEIC)
RUN apk add --no-cache \
      ca-certificates curl tzdata \
      ffmpeg libheif

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