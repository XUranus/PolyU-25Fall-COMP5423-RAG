---
title: Docker Deployment
sidebar_position: 2
description: Step-by-step guide to deploying RAG42 with Docker and Docker Compose.
---

# Docker Deployment

This guide walks you through deploying the full RAG42 stack (backend + frontend) using Docker Compose.

## Prerequisites

Make sure you have the following installed:

- **Docker** (version 20.10+) -- [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** (version 2.0+) -- included with Docker Desktop, or install separately
- **Git** -- to clone the repository
- **Internet connection** -- for downloading images, dependencies, and the BM25 cache

Verify your installation:

```bash
docker --version
docker-compose --version
```

## Step 1: Clone the Repository

```bash
git clone https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG.git
cd PolyU-25Fall-COMP5423-RAG
```

## Step 2: Configure Environment Variables

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```ini title=".env"
RAG42_FRONTEND_PORT=3000
RAG42_BACKEND_PORT=5000
RAG42_BACKEND_HOST=0.0.0.0
RAG42_STORAGE_DIR=/app/storage
RAG42_CACHE_DIR=/app/cache
RAG42_OPENAI_API_KEY=sk-your-api-key-here
RAG42_OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

| Variable | Description | Example |
|----------|-------------|---------|
| `RAG42_FRONTEND_PORT` | Port for the React frontend | `3000` |
| `RAG42_BACKEND_PORT` | Port for the Flask backend | `5000` |
| `RAG42_BACKEND_HOST` | Backend bind address | `0.0.0.0` |
| `RAG42_STORAGE_DIR` | Path inside container for database and logs | `/app/storage` |
| `RAG42_CACHE_DIR` | Path inside container for model/BM25 cache | `/app/cache` |
| `RAG42_OPENAI_API_KEY` | API key for the LLM provider | `sk-xxxx` |
| `RAG42_OPENAI_API_URL` | OpenAI-compatible API endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

:::warning
Never commit your `.env` file with real API keys. The `.gitignore` should exclude it.
:::

## Step 3: Build and Deploy

The easiest way is to use the provided scripts:

```bash
# Build containers and download BM25 cache
bash build.sh

# Start the services
bash deploy.sh
```

### What `build.sh` Does

1. Reads configuration from `.env`
2. Creates `volumes/storage` and `volumes/cache` directories
3. Downloads the pre-built BM25 cache (zip file) to `volumes/cache`
4. Runs `docker-compose build` with the configured ports

```bash title="build.sh (simplified)"
# Create necessary directories
mkdir -p volumes/storage volumes/cache

# Download pre-built BM25 cache
cd volumes/cache
wget https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG/releases/download/BM25Cache/cache.zip
unzip cache.zip

# Build containers
docker-compose build \
    --build-arg RAG42_FRONTEND_PORT=$RAG42_FRONTEND_PORT \
    --build-arg RAG42_BACKEND_PORT=$RAG42_BACKEND_PORT
```

### What `deploy.sh` Does

1. Stops any existing containers (if running)
2. Starts services in detached mode (`-d`)
3. Prints the frontend and backend URLs

```bash title="deploy.sh (simplified)"
# Stop existing containers
docker-compose down

# Start services in background
docker-compose up -d

echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:5000"
```

## Step 4: Verify

Check that both services are running:

```bash
docker-compose ps
```

Expected output:

```
NAME                STATUS          PORTS
rag42-backend       Up (healthy)    0.0.0.0:5000->5000/tcp
rag42-frontend      Up              0.0.0.0:3000->3000/tcp
```

Test the backend health endpoint:

```bash
curl http://localhost:5000/api/health
```

```json
{
  "ok": true,
  "ready": false,
  "storage": "/app/storage",
  "cache": "/app/cache"
}
```

:::note
The `ready` field starts as `false`. It becomes `true` after the RAG pipeline finishes initializing (typically 1-2 minutes). The `/api/chat/<id>/messages` endpoint will not work until `ready` is `true`.
:::

## Docker Compose Configuration

The `docker-compose.yml` defines two services:

```yaml title="docker-compose.yml"
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    env_file:
      - .env
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./volumes/cache:/app/cache
      - ./volumes/storage:/app/storage
    restart: unless-stopped

  frontend:
    env_file:
      - .env
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    restart: unless-stopped
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./volumes/cache` | `/app/cache` | BM25 index cache, downloaded models -- **persists across restarts** |
| `./volumes/storage` | `/app/storage` | SQLite database, log files -- **persists across restarts** |

:::tip
Volume mounts mean your data survives `docker-compose down` and `docker-compose up`. To start fresh, delete the `volumes/` directory.
:::

### Backend Dockerfile

The backend uses a Miniconda base image with a conda environment:

```dockerfile title="backend/Dockerfile"
FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY environment.yml ./
COPY requirements.txt ./
COPY db_init.sql ./
COPY *.py ./

RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "COMP5423-RAG42", "/bin/bash", "-c"]
RUN pip install -r requirements.txt

EXPOSE $RAG42_BACKEND_PORT
CMD ["conda", "run", "-n", "COMP5423-RAG42", "python", "server.py"]
```

## Useful Commands

```bash
# View logs (follow mode)
docker-compose logs -f backend
docker-compose logs -f frontend

# Restart a specific service
docker-compose restart backend

# Stop everything
docker-compose down

# Stop and remove volumes (deletes data!)
docker-compose down -v

# Rebuild after code changes
docker-compose build --no-cache
docker-compose up -d
```

## Troubleshooting

### Port already in use

```
Error: Bind for 0.0.0.0:5000 failed: port is already allocated
```

**Fix:** Change the port in `.env` or stop the process using that port:

```bash
# Find what's using the port
lsof -i :5000
# Kill it, or change RAG42_BACKEND_PORT in .env
```

### BM25 cache download fails

```
wget: unable to resolve host address
```

**Fix:** Check your internet connection. If behind a proxy, configure `http_proxy` and `https_proxy` environment variables. Alternatively, manually download the cache zip from the GitHub releases page and place it in `volumes/cache/`.

### RAG pipeline never becomes ready

The health endpoint keeps showing `"ready": false`.

**Fixes:**
1. Check backend logs: `docker-compose logs backend`
2. Verify your API key is correct in `.env`
3. Ensure the model download completed (first run takes time)
4. Check available memory -- 8 GB minimum, 16 GB recommended

### Container exits immediately

```
rag42-backend exited with code 1
```

**Fix:** Check logs for the error:

```bash
docker-compose logs backend | tail -50
```

Common causes: missing `.env` file, invalid conda environment, or port conflict.

### Frontend cannot reach backend

The frontend loads but API calls fail.

**Fix:** Ensure both containers are running and the backend port matches. Check that `RAG42_BACKEND_PORT` in `.env` matches the port mapping in `docker-compose.yml`.

### Out of disk space

Docker images and volumes can consume significant space.

```bash
# Check Docker disk usage
docker system df

# Clean up unused images and containers
docker system prune
```
