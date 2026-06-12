---
sidebar_position: 2
title: Installation
---

# Installation Guide

This guide walks you through setting up RAG42 from scratch. There are two paths: manual setup (recommended for development) and Docker setup (recommended for deployment).

## Prerequisites

Before you begin, make sure you have the following installed:

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ (tested on 3.12) | Backend runtime |
| Node.js | 18+ | Frontend build and dev server |
| conda | Any recent version | Python environment management |
| pip | Any recent version | Python package installation |
| git | Any recent version | Clone the repository |

:::info GPU is optional
RAG42 was developed and tested on a laptop with an NVIDIA MX150 GPU (2GB VRAM) and 16GB RAM. It works on CPU as well, though generation will be slower.
:::

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

Open `.env` in your editor and fill in the values:

```bash
RAG42_FRONTEND_PORT=3000
RAG42_BACKEND_PORT=5000
RAG42_BACKEND_HOST=0.0.0.0
RAG42_STORAGE_DIR=./volumes/storage
RAG42_CACHE_DIR=./cache
RAG42_OPENAI_API_KEY=sk-your-api-key-here
RAG42_OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

See the [Configuration Reference](./configuration.md) for a full explanation of each variable.

Export the variables into your shell:

```bash
export $(grep -v '^#' .env | xargs)
```

## Step 3: Set Up the Backend

### Create the Conda environment

```bash
cd backend
conda env create -f environment.yml
conda activate COMP5423-RAG42
```

The `environment.yml` creates a Python 3.12 environment with PyTorch 2.3.0, NLTK, and HuggingFace Datasets pre-installed.

### Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, sentence-transformers, FAISS, bm25s, gensim, OpenAI client, and other dependencies.

### Download cached index files

:::warning Critical step -- do not skip!
Without the cached index files, the BM25 indexing process takes **over 3 hours** on first startup. The cache contains pre-built BM25, FAISS, Word2Vec, and ColBERT indices for the HotpotQA subset.
:::

```bash
cd $RAG42_CACHE_DIR
wget https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG/releases/download/BM25Cache/cache.zip
unzip cache.zip
cd ../..
```

After downloading, your cache directory should contain files like:

```
cache/
├── bm25_index_bm25s.npz
├── faiss_index_BAAI_bge-large-en-v1.5.faiss
├── word2vec_model_100d.model
├── colbert_index_colbert-ir_colbertv2.0.npy
├── colbert_index_colbert-ir_colbertv2.0_lengths.npy
└── ...
```

### Start the backend server

```bash
cd backend
python server.py
```

The Flask server starts on `localhost:5000`. On first run it will also download the HotpotQA dataset and (if you selected a local model) the Qwen2.5-0.5B-Instruct model from HuggingFace, which takes about 10 minutes and requires ~1GB of disk space.

:::note
The RAG modules initialize in a background thread. The server is immediately reachable at `/api/health`, but `ready` will be `false` until indexing completes. The frontend polls this endpoint and shows a loading screen.
:::

## Step 4: Set Up the Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm start
```

The React dev server starts on `localhost:3000`. Open that URL in your browser to see the chat interface.

## Docker Setup (Alternative)

:::info Docker is easier for deployment
If you just want to run RAG42 without installing Python or Node.js locally, use Docker. Note that building from scratch takes about 1 hour and the images require ~20GB of disk space.
:::

### Prerequisites

- Docker and Docker Compose installed
- The cache files downloaded into `./volumes/cache/` (see the cache download step above)

### Build and run

```bash
# Create volume directories
mkdir -p volumes/cache volumes/storage

# Download cache if you haven't already
cd volumes/cache
wget https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG/releases/download/BM25Cache/cache.zip
unzip cache.zip
cd ../..

# Build and start
sudo sh build.sh
sudo docker-compose up
```

The `docker-compose.yml` defines two services:

| Service | Port | Description |
|---------|------|-------------|
| `backend` | 5000 | Flask API server (Conda-based) |
| `frontend` | 3000 | React app served by nginx |

Both services read environment variables from the `.env` file.

### Docker volume mounts

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./volumes/cache` | `/app/cache` | Pre-built indices (BM25, FAISS, etc.) |
| `./volumes/storage` | `/app/storage` | SQLite database and logs |

:::warning
You must create the `volumes/cache` directory and download the cache files before running `docker-compose up`. Otherwise the container will try to build indices from scratch, which takes hours.
:::

## Verifying the Installation

After starting both the backend and frontend:

1. Open `http://localhost:3000` in your browser
2. You should see the RAG42 chat interface
3. Wait for the loading screen to disappear (this means the backend is ready)
4. Type a test question like "Who was the director of the movie that won Best Picture at the 2020 Oscars?"
5. You should see the thinking process panel and a final answer

You can also verify the backend directly:

```bash
curl http://localhost:5000/api/health
```

A healthy response looks like:

```json
{
  "ok": true,
  "storage": "./volumes/storage",
  "cache": "./cache",
  "ready": true
}
```

:::note Next steps
Once installed, head to the [Quick Start](./quickstart.md) guide to send your first question.
:::
