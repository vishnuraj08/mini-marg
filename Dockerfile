# ─────────────────────────────────────────────────────────────
# Mini-MARG Dockerfile
#
# WHAT A DOCKERFILE IS:
#   A recipe that tells Docker how to build an image.
#   An image = a frozen snapshot of your app + all dependencies.
#   A container = a running instance of that image.
#
# WHY DOCKER?
#   "Works on my machine" problem solved.
#   The same image runs on your laptop, AWS ECS, or anyone's server.
#   No dependency conflicts, no Python version mismatches.
# ─────────────────────────────────────────────────────────────

# STAGE: Base image
# python:3.11-slim = official Python 3.11 on minimal Debian Linux
# "slim" = no extras, smaller image (~150MB vs ~900MB for full)
# Smaller image = faster pull, less attack surface, lower storage cost
FROM python:3.11-slim

# WHO MAINTAINS THIS IMAGE (metadata only, no functional effect)
LABEL maintainer="yadavvishnuraj@gmail.com"
LABEL description="Mini-MARG: Banking RAG Agent"

# ─────────────────────────────────────────────
# SYSTEM DEPENDENCIES
# ─────────────────────────────────────────────
# FAISS needs these C libraries at runtime
# --no-install-recommends = don't pull in suggested extras (keeps image lean)
# rm -rf /var/lib/apt/lists/* = delete apt cache (reduces image size by ~50MB)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# WORKING DIRECTORY
# ─────────────────────────────────────────────
# All subsequent commands run from /app inside the container
# This is the standard convention for Python web apps
WORKDIR /app

# ─────────────────────────────────────────────
# INSTALL PYTHON DEPENDENCIES
# ─────────────────────────────────────────────
# WHY COPY requirements.txt FIRST (before copying code)?
#   Docker builds in layers. Each instruction = one layer.
#   Layers are CACHED. If requirements.txt hasn't changed,
#   Docker skips the pip install entirely → builds in 5 sec not 5 min.
#   If you copied all code first, any code change = re-install all packages.
#   This ordering is a critical Docker best practice.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_lg
# --no-cache-dir = don't store pip's download cache in the image (saves ~200MB)

# ─────────────────────────────────────────────
# COPY APPLICATION CODE
# ─────────────────────────────────────────────
# Copy everything else AFTER pip install (so code changes don't bust cache)
# .dockerignore (we'll create below) controls what gets excluded
COPY app/         ./app/
COPY data/        ./data/
COPY config/      ./config/
COPY prompts.yaml ./prompts.yaml

# ─────────────────────────────────────────────
# RUNTIME CONFIGURATION
# ─────────────────────────────────────────────
# Port the container listens on (documentation only — doesn't actually publish)
# The actual port mapping happens at `docker run -p 8000:8000`
EXPOSE 8000

# Environment variables
# PYTHONUNBUFFERED=1 → Python prints logs immediately, not buffered
#   Without this: logs appear in batches or not at all in Docker
# PYTHONDONTWRITEBYTECODE=1 → no .pyc files in the container (cleaner)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Ollama URL — the container talks to Ollama on the host machine
# host.docker.internal = Docker's special DNS name for the host machine
# (On Linux use --add-host=host.docker.internal:host-gateway at runtime)
##ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OLLAMA_HOST=http://host.docker.internal:11434

# ─────────────────────────────────────────────
# HEALTHCHECK
# ─────────────────────────────────────────────
# Docker (and AWS ECS) pings this to know if the container is healthy.
# If /health returns non-200 for 3 consecutive checks → container marked unhealthy
# interval=30s  → check every 30 seconds
# timeout=10s   → fail if no response in 10 seconds
# retries=3     → 3 failures before marking unhealthy
# start_period=40s → give app 40 seconds to start before checking
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8000/health || exit 1

# ─────────────────────────────────────────────
# STARTUP COMMAND
# ─────────────────────────────────────────────
# CMD vs ENTRYPOINT:
#   ENTRYPOINT = fixed command (can't be overridden easily)
#   CMD = default command (can be overridden at `docker run`)
#   Best practice for web apps: use CMD
#
# uvicorn app.main:app
#   app.main  = the Python module (app/main.py)
#   :app      = the FastAPI object inside that file
#   --host 0.0.0.0  → listen on ALL network interfaces, not just localhost
#                     Without this: the app is unreachable from outside the container
#   --port 8000     → same port as EXPOSE above
#   --workers 1     → single worker (for simplicity; production uses 2-4)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]