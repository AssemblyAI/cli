# Container image for Fly.io, Railway, Render (Docker), and Cloudflare Containers.
# Vercel ignores this and builds api/index.py as a serverless function instead.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Shell form so ${PORT} expands. Railway/Render inject $PORT; Fly uses its
# fly.toml internal_port (default 8000). Binds 0.0.0.0 so the platform can reach it.
CMD python -m uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8000}
