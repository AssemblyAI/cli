# Container image for Fly.io, Railway, Render (Docker), and Cloudflare Containers.
# Vercel ignores this and builds api/index.py as a serverless function instead.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user (container hardening; the app only reads its files and binds
# a non-privileged port, so it needs no elevated privileges). python:3.13-slim is
# Debian-based, so `useradd` is available.
RUN useradd --create-home appuser
USER appuser

# Fly reads EXPOSE to set its fly.toml internal_port; keep it in sync with the CMD
# default so the proxy and the app agree on the port.
EXPOSE 8080
# Shell form so ${PORT} expands. Railway/Render inject $PORT; Fly maps to 8080.
CMD python -m uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8080}
