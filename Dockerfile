FROM python:3.12-slim

# Environnement
ENV APP_HOME=/home/app
ENV VIRTUAL_ENV=$APP_HOME/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install su-exec
RUN apt-get update && apt-get install -y curl gcc libc-dev && \
    curl -o /usr/local/bin/su-exec.c https://raw.githubusercontent.com/ncopa/su-exec/master/su-exec.c && \
    gcc -Wall /usr/local/bin/su-exec.c -o /usr/local/bin/su-exec && \
    chmod +x /usr/local/bin/su-exec && \
    rm -rf /var/lib/apt/lists/* /usr/local/bin/su-exec.c && \
    apt-get purge -y --auto-remove curl gcc libc-dev

# Crée les répertoires applicatifs
RUN mkdir -p $APP_HOME/static/prints \
    && mkdir -p $APP_HOME/logs \
    && mkdir -p $APP_HOME/data \
    && mkdir -p /var/log/flask-app \
    && touch /var/log/flask-app/flask-app.err.log \
    && touch /var/log/flask-app/flask-app.out.log

WORKDIR $APP_HOME

# Dépendances Python
COPY requirements.txt .
RUN python -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Code applicatif
COPY . .

# Entrée
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
