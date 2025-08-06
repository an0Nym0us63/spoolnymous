# Étape 1 : Builder Tailwind
FROM node:18-bookworm as tailwind-builder

WORKDIR /build

COPY ./tailwind.css ./tailwind.config.js ./
RUN npm install -D tailwindcss postcss autoprefixer \
 && npx tailwindcss -i ./tailwind.css -o ./tailwind.build.css --minify \
 || (echo "🔧 Tailwind build failed — showing tailwind.css contents:" && cat ./tailwind.css && exit 1)

# Étape 2 : Image principale Python
FROM python:3.12.9-slim-bookworm

# permissions and nonroot user for tightened security
RUN adduser --disabled-login nonroot
RUN mkdir /home/app/ && chown -R nonroot:nonroot /home/app
RUN mkdir /home/app/logs/ && chown -R nonroot:nonroot /home/app/logs
RUN mkdir /home/app/data/ && chown -R nonroot:nonroot /home/app/data
RUN mkdir -p /home/app/static/prints && chown -R nonroot:nonroot /home/app/static/prints
RUN mkdir -p /home/app/static/css && chown -R nonroot:nonroot /home/app/static/css
RUN mkdir -p /var/log/flask-app && touch /var/log/flask-app/flask-app.err.log && touch /var/log/flask-app/flask-app.out.log
RUN chown -R nonroot:nonroot /var/log/flask-app

WORKDIR /home/app
USER nonroot

# Copie du projet principal
COPY --chown=nonroot:nonroot . .

# Copie du CSS Tailwind compilé
COPY --from=tailwind-builder /build/tailwind.build.css /home/app/static/css/tailwind.build.css

# Environnement Python
ENV VIRTUAL_ENV=/home/app/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN export FLASK_APP=src/app.py
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:8000", "app:app"]
