FROM python:3.12-slim

# UID/GID configurables (root par défaut)
ARG UID=0
ARG GID=0

# Environnement d'app
ENV APP_HOME=/home/app
ENV VIRTUAL_ENV=$APP_HOME/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Crée l’utilisateur uniquement si UID != 0
RUN if [ "$UID" != "0" ]; then \
      addgroup --gid $GID spooluser && \
      adduser --uid $UID --gid $GID --disabled-login --gecos '' spooluser ; \
    fi

# Prépare les répertoires avec bons droits
RUN mkdir -p $APP_HOME/static/prints \
    && mkdir -p $APP_HOME/logs \
    && mkdir -p $APP_HOME/data \
    && mkdir -p /var/log/flask-app \
    && touch /var/log/flask-app/flask-app.err.log \
    && touch /var/log/flask-app/flask-app.out.log \
    && chown -R $UID:$GID $APP_HOME /var/log/flask-app

WORKDIR $APP_HOME

# Copie des dépendances
COPY requirements.txt .

# VENV + dépendances
RUN python -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code avec bon owner
COPY --chown=$UID:$GID . .

# Expose port HTTP
EXPOSE 8000

# Pas de USER ici → contrôlé côté docker-compose / docker run
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:8000", "src.app:app"]
