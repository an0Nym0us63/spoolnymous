# Stage 1 : Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ .
RUN npm run build

# Stage 2 : Python deps
FROM python:3.12-slim AS py-builder
WORKDIR /install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/pkg -r requirements.txt

# Stage 3 : Image finale
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/an0Nym0us63/spoolnymous"
RUN apt-get update && apt-get install -y --no-install-recommends libssl-dev curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=py-builder /pkg /usr/local
COPY backend/ ./
COPY --from=frontend-builder /frontend/dist ./static/
ARG BUILD_BRANCH=v2
ARG COMMIT_SHA=unknown
ARG BUILD_DATE=unknown
ENV BUILD_BRANCH=${BUILD_BRANCH} COMMIT_SHA=${COMMIT_SHA} BUILD_DATE=${BUILD_DATE}
ENV DATA_DIR=/data DATABASE_URL=sqlite+aiosqlite:////data/spoolnymous.db UPLOADS_DIR=/data/uploads
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD curl -f http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
