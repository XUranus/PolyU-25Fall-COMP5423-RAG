---
sidebar_position: 4
title: Configuration
---

# Configuration Reference

RAG42 is configured through environment variables. This page documents every variable, the cache directory structure, and the `.env.example` file.

## Environment Variables

All variables use the `RAG42_` prefix. Set them in a `.env` file at the project root, then export them:

```bash
export $(grep -v '^#' .env | xargs)
```

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG42_FRONTEND_PORT` | `3000` | Port for the React dev server (or nginx in Docker) |
| `RAG42_BACKEND_PORT` | `5000` | Port for the Flask backend API server |
| `RAG42_BACKEND_HOST` | `0.0.0.0` | Host address the Flask server binds to. Use `0.0.0.0` to accept connections from any interface, or `127.0.0.1` for localhost only |
| `RAG42_CORS_ORIGINS` | `*` | Comma-separated list of allowed CORS origins, or `*` for all. Example: `http://localhost:3000,https://myapp.com` |

### Storage Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG42_STORAGE_DIR` | `./storage` | Directory for persistent data: the SQLite database (`chat_history.db`) and the log file (`rag.log`). Created automatically if it does not exist |
| `RAG42_CACHE_DIR` | `./cache` | Directory for cached index files (BM25, FAISS, Word2Vec, ColBERT). Created automatically. **At least 1GB of free space is recommended** |

:::tip
In Docker deployments, these paths are typically set to `/app/storage` and `/app/cache`, which are mounted as Docker volumes. See the `docker-compose.yml` for the volume mappings.
:::

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG42_OPENAI_API_KEY` | *(none)* | API key for the OpenAI-compatible service. Required only if you use an API-based model (not needed for local Qwen models) |
| `RAG42_OPENAI_API_URL` | *(none)* | Base URL for the OpenAI-compatible API endpoint. For example, `https://dashscope.aliyuncs.com/compatible-mode/v1` for Alibaba DashScope, or `https://api.openai.com/v1` for OpenAI |

:::warning
If you use a remote LLM via the OpenAI API, both `RAG42_OPENAI_API_KEY` and `RAG42_OPENAI_API_URL` must be set. If either is missing, the `OpenAIGenerator` will raise an assertion error at initialization.
:::

### Cache Directory Structure

The `$RAG42_CACHE_DIR` directory stores pre-built indices so the server starts quickly. Without these files, the server builds them from scratch on first startup (taking 3+ hours for BM25 alone).

```
$RAG42_CACHE_DIR/
├── bm25_index_bm25s.npz                          # BM25 sparse index (bm25s library)
├── faiss_index_BAAI_bge-large-en-v1.5.faiss       # FAISS dense index (BGE embeddings)
├── word2vec_model_100d.model                      # Word2Vec model (gensim)
├── colbert_index_colbert-ir_colbertv2.0.npy       # ColBERT token-level embeddings
├── colbert_index_colbert-ir_colbertv2.0_lengths.npy  # ColBERT document lengths
└── ...
```

| File | Created By | Size | Description |
|------|-----------|------|-------------|
| `bm25_index_bm25s.npz` | `SparseRetriever` | ~50MB | BM25 inverted index built by the `bm25s` library. Tokenized with English stopword removal |
| `faiss_index_BAAI_bge-large-en-v1.5.faiss` | `DenseRetriever` | ~500MB | FAISS `IndexFlatIP` index of document embeddings from `BAAI/bge-large-en-v1.5`. Normalized for cosine similarity |
| `word2vec_model_100d.model` | `StaticEmbeddingRetriever` | ~10MB | A 100-dimensional Word2Vec model trained on the document corpus using gensim |
| `colbert_index_*.npy` | `ColBERTRetriever` | ~200MB | NumPy arrays of token-level embeddings from `colbert-ir/colbertv2.0` |

:::info
You can delete individual cache files to save disk space. The system will rebuild only the missing indices on next startup. For example, if you only use the hybrid retriever (BM25 + BGE), you can safely delete the Word2Vec and ColBERT files.
:::

## Example .env File

Here is the complete `.env.example` from the repository:

```bash
# Frontend port (React dev server or nginx)
RAG42_FRONTEND_PORT=3000

# Backend port (Flask API server)
RAG42_BACKEND_PORT=5000

# Backend host binding address
RAG42_BACKEND_HOST=0.0.0.0

# Directory for database and log files
RAG42_STORAGE_DIR=/app/storage

# Directory for cached index files (BM25, FAISS, etc.)
RAG42_CACHE_DIR=/app/cache

# OpenAI-compatible API key (required for remote LLMs)
RAG42_OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI-compatible API base URL
RAG42_OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

:::note
The `.env.example` uses Docker-style paths (`/app/storage`, `/app/cache`). For local development, change these to relative paths like `./volumes/storage` and `./cache`.
:::

## Frontend Configuration

The frontend has a separate configuration file at `frontend/src/config.ts` that controls which LLM models appear in the dropdown. This file is compiled into the React build and does not use environment variables at runtime.

:::tip
To add a new model to the frontend dropdown, edit `frontend/src/config.ts` and rebuild the frontend with `npm start` (dev mode picks up changes automatically).
:::

## Database Schema

RAG42 uses SQLite for chat history. The database file is created at `$RAG42_STORAGE_DIR/chat_history.db`. The schema has two tables:

**`chat_sessions`** -- stores chat sessions

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) | Primary key |
| `title` | TEXT | Chat title (auto-set from the first message) |
| `created_at` | DATETIME | Creation timestamp |
| `updated_at` | DATETIME | Last update timestamp |

**`messages`** -- stores individual messages

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) | Primary key |
| `session_id` | TEXT (UUID) | Foreign key to `chat_sessions.id` |
| `sender` | TEXT | Either `'user'` or `'bot'` |
| `content` | TEXT | The message text |
| `timestamp` | DATETIME | Message timestamp |
| `thinking_process` | TEXT | JSON array of reasoning steps (only for bot messages) |

:::note
The database is initialized from `backend/db_init.sql` on first startup. Deleting the database file resets all chat history.
:::
