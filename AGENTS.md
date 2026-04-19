# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

RAG42 is a Retrieval-Augmented Generation system for multi-hop question answering on the HotpotQA dataset. It has a Python/Flask backend with the RAG pipeline, a React/TypeScript frontend, and evaluation scripts. Course project for PolyU COMP5423.

The evaluation uses two metrics: **nDCG@10** for retrieval quality and **EM/F1** for answer accuracy. Both contribute proportionally to the final score and ranking.

## Commands

### Backend

```bash
cd backend
conda env create -f environment.yml          # Create conda env (name: COMP5423-RAG42)
conda activate COMP5423-RAG42
pip install -r requirements.txt
python server.py                             # Start Flask server on localhost:5000
```

To speed up first startup, download pre-built BM25 cache:
```bash
cd $RAG42_CACHE_DIR
wget https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG/releases/download/BM25Cache/cache.zip
unzip cache.zip
```

### Frontend

```bash
cd frontend
npm install
npm start                                    # Dev server on localhost:3000
npm run build                                # Production build
npm test                                     # Run React tests
```

### Evaluation

```bash
cd evaluate
python test_predict.py -d <dataset> -g <model> -r <retriever>  # Run predictions
python eval_retrieval.py --gold <gold> --pred <pred>             # Retrieval metrics (MAP, NDCG, Recall)
python eval_hotpotqa.py --gold <gold> --pred <pred>              # QA metrics (EM, F1, joint EM/F1)
```

Retriever options: `sparse`, `static_embedding`, `dense`, `instruction`, `colbert`, `hybrid`
Generator options: `qwen2.5-0.5b-instruct`, `qwen2.5-1.5b-instruct`, `qwen2.5-3b-instruct`, `qwen2.5-7b-instruct`

### Docker

```bash
cp .env.example .env && export $(grep -v '^#' .env | xargs)
sudo sh build.sh                             # Build images + download cache
sudo docker-compose up                       # Run both services
```

## Environment Variables

All prefixed with `RAG42_`. Required: `RAG42_OPENAI_API_KEY`, `RAG42_OPENAI_API_URL`. See `.env.example` for full list. The backend reads them via `os.getenv()` with fallback defaults.

## Architecture

### Request Flow

Frontend sends `POST /api/chat/{chatId}/messages` with `{message, model_name}`. Flask stores the message in SQLite, runs the RAG pipeline, and returns the answer with a `thinking_process` log. The frontend renders the answer and an expandable thinking panel.

### Backend RAG Pipeline

Entry point: `rag_pipeline.py` -> `RAGPipeline.run()`. Selects generator based on `model_name`, then delegates to a workflow.

**Workflows:**
- `AgenticWorkflow` (default): Uses LLM to classify questions as multi-hop or single-hop. For multi-hop, decomposes into sub-questions, retrieves and answers each sub-question, then synthesizes a final answer. Falls back to heuristic regex if LLM classification fails.
- `SingleHopWorkflow` (baseline): Single retrieve-then-generate, no decomposition.

**Retriever hierarchy** (factory method in `retriever_base.py`):
- `SparseRetriever` -- BM25 via `bm25s`, cached as `.npz`
- `DenseRetriever` -- BAAI/bge-large-en-v1.5 embeddings + FAISS IndexFlatIP, cached as `.faiss`. Uses BGE query prefix `"Represent this sentence: "` for queries.
- `StaticEmbeddingRetriever` -- gensim Word2Vec, cached as `.model`
- `InstructionRetriever` -- E5-instruct with task-specific instruction prefix, cached as `.faiss`. Covers the "Dense retrieval with instruction" category.
- `ColBERTRetriever` -- Multi-vector retrieval using ColBERT MaxSim scoring, cached as `.npy`. Covers the "Multi-vector retrieval" category.
- `HybridRetriever` -- RRF (Reciprocal Rank Fusion) of sparse + dense, with optional cross-encoder re-ranking via `BAAI/bge-reranker-v2-m3`

**Re-ranker** (`reranker.py`):
- `CrossEncoderReranker` -- Cross-encoder re-scoring of top RRF candidates. Enabled by default in `HybridRetriever`.

**Generators:**
- `HuggingfaceGenerator` -- Only for `Qwen/Qwen2.5-0.5B-Instruct` (local inference)
- `OpenAIGenerator` -- All other models via OpenAI-compatible API (Aliyun DashScope)

**Answer post-processing** (`agentic_workflow.py:_post_process_answer`):
Strips verbose prefixes ("The answer is:", "Final Answer:", etc.), takes first non-empty line, removes trailing period. This normalizes LLM output to the short entity format HotpotQA expects.

### Database

SQLite with two tables: `chat_sessions` (id, title, timestamps) and `messages` (id, session_id FK, sender, content, timestamp, thinking_process JSON). Schema in `backend/db_init.sql`.

### Frontend

React 19 + TypeScript + TailwindCSS (dark theme). Component tree: `App` -> `ChatPage` -> (`InitPage` | `Sidebar` + `ChatPanel` -> `ThinkingPanel` + `LoadingButton`). `InitPage` polls `/api/health` until the backend is ready. `ThinkingPanel` renders step logs with markdown. Nginx serves the production build and proxies `/api/` to Flask.

### Async Initialization

RAG modules (retriever index building, model loading) run in a background daemon thread on server startup. The frontend polls `/api/health` which returns `ready: true` once initialization completes.

## Conventions

- All backend modules use `logging.getLogger('RAG42')` as the shared logger.
- Retriever indices cache to `$RAG42_CACHE_DIR`. Pre-built BM25 cache can be downloaded from GitHub releases.
- Chat sessions and messages use UUID v4 identifiers.
- No linting or formatting configuration is present in this project.
