# PolyU-25Fall-COMP5423-RAG
RAG Project for COMP5423 of PolyU 25Fall 

## Build
To build a docker image:
```bash
cd $path_to_project
sh build.sh
```

## Environments
Applicable environments. e.g,
```
RAG42_FRONTEND_PORT=3000
RAG42_BACKEND_PORT=5000
RAG42_BACKEND_HOST=0.0.0.0
RAG42_STORAGE_DIR=./volumes/storage
RAG42_CACHE_DIR=./cache
RAG42_OPENAI_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
RAG42_OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```