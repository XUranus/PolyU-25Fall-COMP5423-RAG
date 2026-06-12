# Changelog

## [Stage 4] - 2026-04-19

### Code Quality & Consistency

- **Changed `HybridRetriever` to `BaseRetriever` type hints**: Both `AgenticWorkflow` and `SingleHopWorkflow` now use `BaseRetriever` as the type hint for the retriever parameter, making them compatible with all retriever types (sparse, dense, hybrid, instruction, colbert) instead of being locked to `HybridRetriever` (`agentic_workflow.py`, `singlehop_workflow.py`)
- **Included `result` field for all generation steps in thinking_process**: Previously only `multi_hop_sub_generation` steps included the `result` field. Now all steps that have a `result` key (including `single_hop_generation`, `multi_hop_synthesize_answer`) are included in the thinking process output, improving the UI's ability to display intermediate answers (`rag_pipeline.py`)

### Evidence Truncation Improvement

- **Replaced per-document character truncation with total character budget**: `answer_from_docs` now uses `max_total_chars=8000` (~2000 tokens) as a total evidence budget instead of `max_doc_chars=2000` per document. This prevents context overflow when there are many documents, and allocates more space to shorter docs while capping the total. Both `AgenticWorkflow` and `SingleHopWorkflow` use the same approach (`agentic_workflow.py`, `singlehop_workflow.py`)

## [Stage 3] - 2026-04-19

### Answer Verification & Multi-hop Retrieval

- **Added answer verification**: New `verify_answer` method uses the LLM to check if the generated answer is supported by the evidence. Single-hop answers now go through `answer_with_verification` which retries once if verification fails (`agentic_workflow.py`)
- **Retrieve for original question in multi-hop path**: Added a retrieval step for the original question alongside sub-question retrievals. Some supporting documents may only be found via the full original question, not the decomposed sub-questions (`agentic_workflow.py`)

### Thread Safety & Robustness

- **Fixed thread safety in `RAGPipeline.init_generator`**: Added `threading.Lock` with double-check locking pattern to prevent race conditions when multiple threads initialize the same generator simultaneously (`rag_pipeline.py`)
- **Fixed `test_predict.py` append mode**: Output file now opens in write mode (`"w"`) and existing file is explicitly removed before execution, preventing stale results from accumulating. Reduced `max_workers` from 20 to 4 to avoid overwhelming the API. Pre-initializes the generator before the thread pool (`test_predict.py`)

## [Stage 2] - 2026-04-19

### Workflow Improvements

- **Skip LLM classification, always attempt decomposition**: Since HotpotQA is a multi-hop dataset, the LLM-based multi-hop classification step was removed. The workflow now always attempts query decomposition and falls back to single-hop only if decomposition yields <2 sub-questions. This saves 1 API call per question and ensures no multi-hop question is missed (`agentic_workflow.py`)
- **Chain reasoning for sub-questions**: `answer_from_docs` now accepts `prior_answers` parameter. When processing sub-question N, the answers from sub-questions 1..N-1 are injected into the prompt as context. This enables the LLM to use intermediate results when answering dependent sub-questions (e.g., "Who directed [answer from 1]?") (`agentic_workflow.py`)

### Bug Fixes

- **Fixed SingleHopWorkflow prompt**: Replaced broken prompt with stray `"nContext:"` typo with the improved prompt format matching `AgenticWorkflow`. Added `_post_process_answer` method to SingleHopWorkflow for consistent answer normalization (`singlehop_workflow.py`)
- **Fixed HuggingfaceGenerator system prompt**: Updated from generic `"helpful assistant"` to `"precise question-answering assistant"` matching the OpenAIGenerator (`huggingface_generator.py`)
- **Fixed broken f-strings in `rag_pipeline.py`**: `logger.info("start init new generator: {model_name}")` and `logger.info("new generator: {model_name} loaded.")` were missing `f` prefix, causing literal `{model_name}` to be printed instead of the actual model name (`rag_pipeline.py`)

## [Stage 1] - 2026-04-19

### Retrieval Improvements

- **Upgraded default dense model** from `BAAI/bge-small-en-v1.5` (33M params) to `BAAI/bge-large-en-v1.5` (335M params) for significantly better retrieval quality (`dense_retriever.py`)
- **Added BGE query prefix** `"Represent this sentence: "` to dense retrieval queries, as recommended by the BGE model authors (`dense_retriever.py`)
- **Replaced weighted score fusion with Reciprocal Rank Fusion (RRF)** in `HybridRetriever`. RRF is more robust than min-max normalized weighted sum and does not require alpha tuning (`hybrid_retriever.py`)
- **Added cross-encoder re-ranker** (`reranker.py`, `BAAI/bge-reranker-v2-m3`). Re-scores top RRF candidates for more accurate relevance ranking. Enabled by default in `HybridRetriever`
- **Added instruction-based dense retriever** (`instruction_retriever.py`, `intfloat/multilingual-e5-large-instruct`). Supports task-specific instruction prefixes for queries. Covers the "Dense retrieval with instruction" category for full implementation grade
- **Added ColBERT multi-vector retriever** (`colbert_retriever.py`). Uses MaxSim late interaction scoring with token-level embeddings. Covers the "Multi-vector retrieval" category for full implementation grade
- **Registered new retrievers** (`instruction`, `colbert`) in the `BaseRetriever.create_retriever` factory method (`retriever_base.py`)

### Generation & Answer Quality Improvements

- **Replaced regex-only multi-hop detection with LLM-based classification** in `AgenticWorkflow`. The LLM determines if a question is multi-hop; falls back to heuristic regex on error; defaults to multi-hop for ambiguous responses (appropriate for HotpotQA) (`agentic_workflow.py`)
- **Improved answer extraction prompts**: both `answer_from_docs` and `synthesize_answer` now explicitly instruct the LLM to produce short entity-style answers (`"Your answer must be a short phrase, a single entity name, or 'yes'/'no'"`) (`agentic_workflow.py`)
- **Added `_post_process_answer`** method that strips common verbose prefixes (`"The answer is:"`, `"Final Answer:"`, etc.), takes the first non-empty line, and removes trailing periods (`agentic_workflow.py`)
- **Updated OpenAI generator system prompt** from generic `"helpful assistant"` to `"precise question-answering assistant"` that produces short direct answers. Fixed message order (system before user) (`openai_generator.py`)

### Evaluation Pipeline Fix

- **Fixed `retrieved_docs` collection in `test_predict.py`**: now keeps the highest score per document across all steps (instead of `extend` which could include duplicates) and takes exactly top 10 by score

### Documentation

- **Created `AGENTS.md`** with project overview, commands, architecture, and conventions for future Qoder instances
- **Created `docs/CHANGELOG.md`** to track project changes
