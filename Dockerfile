# Use an official Python runtime as a parent image
FROM python:3.12.9-slim-bookworm

# permissions and nonroot user for tightened security
RUN adduser --disabled-login nonroot
RUN mkdir /home/app/ && chown -R nonroot:nonroot /home/app
RUN mkdir /home/app/logs/ && chown -R nonroot:nonroot /home/app/logs
RUN mkdir /home/app/data/ && chown -R nonroot:nonroot /home/app/data
RUN mkdir -p /home/app/static/prints && chown -R nonroot:nonroot /home/app/static/prints
RUN mkdir -p /var/log/flask-app && touch /var/log/flask-app/flask-app.err.log && touch /var/log/flask-app/flask-app.out.log
RUN chown -R nonroot:nonroot /var/log/flask-app
WORKDIR /home/app
USER nonroot
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
    rm -rf /var/lib/apt/lists/*
# Télécharge go2rtc
RUN curl -L -o /home/app/go2rtc https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_linux_amd64 && \
    chmod +x /home/app/go2rtc
# copy all the files to the container
COPY --chown=nonroot:nonroot . .

# venv
ENV VIRTUAL_ENV=/home/app/venv

# python setup
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN export FLASK_APP=src/app.py
RUN pip install --no-cache-dir -r requirements.txt

# define the port number the container should expose
EXPOSE 8000

CMD ["sh", "-c", "python generate_go2rtc_config.py && ./go2rtc & exec gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 app:app"]
