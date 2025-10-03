FROM python:3.12-slim AS app

ENV APP_HOME=/home/app
ENV VIRTUAL_ENV=$APP_HOME/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Paquets système: ffmpeg + libheif pour HEIC, gosu si tu veux dropper root
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libheif1 ca-certificates curl tzdata gosu \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root
RUN useradd -u 1000 -m app \
 && mkdir -p $APP_HOME/static/prints $APP_HOME/static/uploads $APP_HOME/logs /var/log/flask-app \
 && touch /var/log/flask-app/flask-app.err.log /var/log/flask-app/flask-app.out.log \
 && chown -R app:app $APP_HOME /var/log/flask-app

WORKDIR $APP_HOME

# Dépendances Python (wheels dispo sur 3.12)
COPY --chown=app:app requirements.txt .
RUN python -m venv $VIRTUAL_ENV \
 && pip install --upgrade pip wheel setuptools \
 && pip install --no-cache-dir -r requirements.txt

# Code
COPY --chown=app:app . .

# Entrée
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
# Si ton entrypoint utilise su-exec, remplace-le par gosu, ou lance direct en USER app :
USER app
ENTRYPOINT ["/entrypoint.sh"]
