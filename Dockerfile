FROM python:3.12-slim

# Paramètres UID/GID configurables (root par défaut)
ARG UID=0
ARG GID=0

# Définition des chemins
ENV APP_HOME=/home/app
ENV VIRTUAL_ENV=$APP_HOME/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Création groupe et utilisateur avec UID/GID paramétrables
RUN addgroup --gid ${GID} spooluser && \
    adduser --uid ${UID} --gid ${GID} --disabled-login --gecos '' spooluser

# Création des répertoires nécessaires
RUN mkdir -p $APP_HOME/static/prints \
    && mkdir -p $APP_HOME/logs \
    && mkdir -p $APP_HOME/data \
    && mkdir -p /var/log/flask-app \
    && touch /var/log/flask-app/flask-app.err.log \
    && touch /var/log/flask-app/flask-app.out.log \
    && chown -R ${UID}:${GID} $APP_HOME /var/log/flask-app

# Définir le dossier de travail
WORKDIR $APP_HOME

# Copier les fichiers de dépendances en premier
COPY requirements.txt .

# Création de l’environnement virtuel et installation des dépendances
RUN python -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source avec les bons UID/GID
COPY --chown=${UID}:${GID} . .

# Exposer le port HTTP
EXPOSE 8000

# Pas de USER ici — il sera défini à l'exécution via docker run / compose
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:8000", "src.app:app"]
