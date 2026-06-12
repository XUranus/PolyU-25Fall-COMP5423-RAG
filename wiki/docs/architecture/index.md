---
sidebar_position: 1
title: System Overview
---

# System Architecture Overview

This page describes the internal architecture of RAG42: the directory structure, module relationships, class hierarchies, and design patterns.

## Directory Structure

```
PolyU-25Fall-COMP5423-RAG/
├── backend/
│   ├── server.py                  # Flask API server (routes, DB, async init)
│   ├── rag_pipeline.py            # Main orchestrator: ties retrieval + generation
│   ├── agentic_workflow.py        # Multi-hop decompose → retrieve → generate → synthesize
│   ├── singlehop_workflow.py      # Simple single-hop RAG baseline
│   ├── retriever_base.py          # BaseRetriever ABC + factory method
│   ├── sparse_retriever.py        # BM25 retrieval (bm25s library)
│   ├── dense_retriever.py         # BGE dense retrieval (Sentence Transformers + FAISS)
│   ├── static_embedding_retriever.py  # Word2Vec retrieval (gensim)
│   ├── instruction_retriever.py   # E5-instruct dense retrieval
│   ├── colbert_retriever.py       # ColBERT multi-vector retrieval
│   ├── hybrid_retriever.py        # BM25 + BGE + RRF fusion + reranker
│   ├── reranker.py                # Cross-encoder re-ranking (bge-reranker-v2-m3)
│   ├── generator_base.py          # BaseGenerator ABC
│   ├── huggingface_generator.py   # Local Qwen model via transformers
│   ├── openai_generator.py        # Remote LLM via OpenAI-compatible API
│   ├── rag_utils.py               # Shared utilities (answer post-processing, evidence building)
│   ├── db_init.sql                # SQLite schema initialization
│   ├── environment.yml            # Conda environment definition
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # Backend Docker image
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Root component with routing
│   │   ├── ChatPage.tsx           # Main chat page layout
│   │   ├── InitPage.tsx           # Loading screen while backend initializes
│   │   ├── config.ts              # Model list and API URL configuration
│   │   └── components/
│   │       ├── ChatPanel.tsx      # Message list and input box
│   │       ├── Sidebar.tsx        # Chat session list
│   │       ├── ThinkingPanel.tsx  # Reasoning steps visualization
│   │       └── LoadingButton.tsx  # Animated loading button
│   ├── package.json
│   └── Dockerfile
├── evaluate/
│   ├── eval_hotpotqa.py           # Answer evaluation (F1, EM, pytrec_eval)
│   ├── eval_retrieval.py          # Retrieval evaluation (recall, precision)
│   └── test_predict.py            # Batch prediction script
├── .env.example
├── docker-compose.yml
├── build.sh
└── deploy.sh
```

## Module Dependency Diagram

The following diagram shows how the backend modules depend on each other:

```mermaid
graph TD
    server["server.py<br/>(Flask API)"]
    pipeline["rag_pipeline.py<br/>(RAGPipeline)"]
    agentic["agentic_workflow.py<br/>(AgenticWorkflow)"]
    singlehop["singlehop_workflow.py<br/>(SingleHopWorkflow)"]
    retriever_base["retriever_base.py<br/>(BaseRetriever)"]
    hybrid["hybrid_retriever.py<br/>(HybridRetriever)"]
    sparse["sparse_retriever.py<br/>(SparseRetriever)"]
    dense["dense_retriever.py<br/>(DenseRetriever)"]
    static_emb["static_embedding_retriever.py<br/>(StaticEmbeddingRetriever)"]
    instruction["instruction_retriever.py<br/>(InstructionRetriever)"]
    colbert["colbert_retriever.py<br/>(ColBERTRetriever)"]
    reranker["reranker.py<br/>(CrossEncoderReranker)"]
    generator_base["generator_base.py<br/>(BaseGenerator)"]
    hf_gen["huggingface_generator.py<br/>(HuggingfaceGenerator)"]
    openai_gen["openai_generator.py<br/>(OpenAIGenerator)"]
    utils["rag_utils.py<br/>(Utilities)"]

    server --> pipeline
    pipeline --> agentic
    pipeline --> singlehop
    pipeline --> generator_base
    agentic --> retriever_base
    agentic --> utils
    singlehop --> retriever_base
    singlehop --> utils
    hybrid --> sparse
    hybrid --> dense
    hybrid --> reranker
    sparse --> retriever_base
    dense --> retriever_base
    static_emb --> retriever_base
    instruction --> retriever_base
    colbert --> retriever_base
    hf_gen --> generator_base
    openai_gen --> generator_base
```

## Class Diagram: Retrievers

All retrievers inherit from `BaseRetriever` and implement the `retrieve()` method:

```mermaid
classDiagram
    class BaseRetriever {
        <<abstract>>
        +collection_path: str
        +cache_dir: str
        +doc_texts: List[str]
        +doc_ids: List[str]
        +id_to_text: Dict[str, str]
        +retrieve(query, k) List~Tuple~*abstract*
        +create_retriever(type, kwargs) BaseRetriever$
        -_load_collection(path)
    }

    class SparseRetriever {
        +bm25_retriever: BM25
        +retrieve(query, k)
        -_build_index()
    }

    class DenseRetriever {
        +dense_model: SentenceTransformer
        +dense_index: FaissIndex
        +retrieve(query, k)
        -_build_index()
    }

    class StaticEmbeddingRetriever {
        +word2vec_model: Word2Vec
        +doc_embeddings: ndarray
        +retrieve(query, k)
        -_build_index()
        -_compute_document_embeddings()
    }

    class InstructionRetriever {
        +dense_model: SentenceTransformer
        +dense_index: FaissIndex
        +query_instruction: str
        +retrieve(query, k)
        -_build_index()
    }

    class ColBERTRetriever {
        +model: SentenceTransformer
        +doc_embeddings: List
        +retrieve(query, k)
        -_build_index()
        -_maxsim_score()
    }

    class HybridRetriever {
        +sparse_retriever: SparseRetriever
        +dense_retriever: DenseRetriever
        +reranker: CrossEncoderReranker
        +rrf_k: int
        +retrieve(query, k)
    }

    BaseRetriever <|-- SparseRetriever
    BaseRetriever <|-- DenseRetriever
    BaseRetriever <|-- StaticEmbeddingRetriever
    BaseRetriever <|-- InstructionRetriever
    BaseRetriever <|-- ColBERTRetriever
    BaseRetriever <|-- HybridRetriever
    HybridRetriever --> SparseRetriever : uses
    HybridRetriever --> DenseRetriever : uses
    HybridRetriever --> CrossEncoderReranker : uses
```

## Class Diagram: Generators

Generators follow a simple abstract interface:

```mermaid
classDiagram
    class BaseGenerator {
        <<abstract>>
        +generate(prompt) str*
    }

    class HuggingfaceGenerator {
        +model_name: str
        +model: AutoModelForCausalLM
        +tokenizer: AutoTokenizer
        +generate(prompt) str
    }

    class OpenAIGenerator {
        +model_name: str
        +client: OpenAI
        +api_key: str
        +api_url: str
        +generate(prompt) str
    }

    BaseGenerator <|-- HuggingfaceGenerator
    BaseGenerator <|-- OpenAIGenerator
```

## Key Abstractions

### BaseRetriever (Abstract Base Class)

All retrievers extend `BaseRetriever`. This class handles loading the document collection and defines the `retrieve()` contract:

```python
# backend/retriever_base.py
class BaseRetriever(ABC):
    def __init__(self, collection_path: str, cache_dir: str, skip_load: bool = False):
        self.collection_path = collection_path
        self.cache_dir = cache_dir
        if not skip_load:
            self._load_collection(collection_path)

    @abstractmethod
    def retrieve(self, query: str, k: int = 20) -> List[Tuple[str, str, float]]:
        """Returns list of (doc_id, doc_text, score) tuples."""
        pass

    @classmethod
    def create_retriever(cls, retriever_type: str, **kwargs):
        """Factory method: 'sparse', 'dense', 'hybrid', etc."""
        ...
```

### BaseGenerator (Abstract Base Class)

Generators implement a single `generate()` method:

```python
# backend/generator_base.py
class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Takes a prompt string, returns the generated response."""
        pass
```

## Design Patterns

RAG42 uses several well-known design patterns:

### 1. Abstract Base Class (ABC)

Both `BaseRetriever` and `BaseGenerator` are abstract classes that define a contract. Every retriever must implement `retrieve()`, and every generator must implement `generate()`. This makes it easy to add new retrieval or generation strategies without changing the rest of the system.

### 2. Factory Method

`BaseRetriever.create_retriever()` is a factory method that creates the right retriever subclass based on a string type:

```python
retriever = BaseRetriever.create_retriever("hybrid", collection_path="izhx/COMP5423-25Fall-HQ-small")
```

### 3. Strategy Pattern

The `RAGPipeline` treats the retriever and generator as interchangeable strategies. You can swap `HybridRetriever` for `SparseRetriever`, or `HuggingfaceGenerator` for `OpenAIGenerator`, without changing the pipeline logic.

### 4. Template Method

The `AgenticWorkflow.run()` method defines the overall algorithm skeleton (decompose, retrieve, generate per sub-question, synthesize), while the individual steps (how to decompose, how to retrieve) are implemented as separate methods that can be overridden.

### 5. Lazy Initialization with Double-Checked Locking

Generators are initialized on first use in `RAGPipeline.init_generator()`. The method uses a lock to ensure thread safety when multiple requests arrive before the generator is ready:

```python
def init_generator(self, model_name: str):
    if model_name in self.generator_map:
        return self.generator_map[model_name]
    with self._generator_lock:
        if model_name in self.generator_map:  # double-check
            return self.generator_map[model_name]
        # ... initialize generator ...
```

## Component Responsibilities

| Component | File | Responsibility |
|-----------|------|---------------|
| **Flask Server** | `server.py` | HTTP API, SQLite persistence, async RAG initialization |
| **RAG Pipeline** | `rag_pipeline.py` | Orchestrates retrieval + generation, manages generator lifecycle |
| **Agentic Workflow** | `agentic_workflow.py` | Multi-hop decomposition, chain reasoning, answer verification |
| **SingleHop Workflow** | `singlehop_workflow.py` | Simple single-step RAG (baseline) |
| **Hybrid Retriever** | `hybrid_retriever.py` | BM25 + BGE fusion with RRF and cross-encoder re-ranking |
| **Sparse Retriever** | `sparse_retriever.py` | BM25 retrieval via bm25s |
| **Dense Retriever** | `dense_retriever.py` | BGE embedding retrieval via FAISS |
| **Cross-Encoder** | `reranker.py` | Re-scores candidate documents using bge-reranker-v2-m3 |
| **HuggingFace Generator** | `huggingface_generator.py` | Local LLM inference (Qwen2.5) |
| **OpenAI Generator** | `openai_generator.py` | Remote LLM inference via OpenAI-compatible API |
| **RAG Utils** | `rag_utils.py` | Answer post-processing, evidence snippet building |

:::note Next steps
Continue to [Data Flow](./data-flow.md) to see how a single request moves through the entire system.
:::
