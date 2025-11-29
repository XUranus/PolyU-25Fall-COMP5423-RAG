# PolyU-25Fall-COMP5423-RAG
> RAG Project for COMP5423 of PolyU 25Fall 

## Objective
This project aims to develop a **Retrieval-Augmented Generation (RAG)** system using a subset of the HotpotQA dataset, which contains multi-hop, Wikipedia-based questions along with their corresponding answers. The objective is to build an end-to-end question answering system capable of handling complex queries that require reasoning across multiple documents. By integrating a retrieval module with a generative large language model, the system will improve both the factual accuracy and explainability of answers through evidencegrounded response generation. The HotpotQA dataset will serve as the training and evaluation foundation, with a focus on understanding the application of RAG techniques to multi-hop, explainable question-answering tasks. The project will optimize both retrieval and generation components to enhance overall system performance for real-world RAG applications.

## Project Resource
In this project, we will develop and evaluate an RAG system based on a sampled subset of `HotpotQA`. The `Train` and `Validation` splits with relevant documents could be used in developing and self evaluation. The sampled `HQ-small` is released at [https://huggingface.co/datasets/izhx/COMP5423-25Fall-HQ-small](https://huggingface.co/datasets/izhx/COMP5423-25Fall-HQ-small)

![](screenshot.png)

## Build & Run
Copy `.env.example` to `.env` and configure it with your preferences. 
Applicable environments in `.env` are like:
```bash
RAG42_FRONTEND_PORT=3000
RAG42_BACKEND_PORT=5000
RAG42_BACKEND_HOST=0.0.0.0
RAG42_STORAGE_DIR=./volumes/storage # for database and logs storage
RAG42_CACHE_DIR=./cache # Spare at least 1GB for cache storage
RAG42_OPENAI_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx # OpenAI API Key
RAG42_OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 # OpenAI API Sever
```
Then run `export $(grep -v '^#' .env | xargs)` to export the variables.

### Using Docker
Start to build images and to run:
```bash
sudo sh build.sh
sudo docker-compose up
```

### Build And Run Manually
1. Prepare backend server environment
    ```bash
    cd backend
    conda env create -f environment.yml
    conda activate COMP5423-RAG42
    pip install -r requirements.txt
    ```
    You can download the cached index file to the `$RAG42_CACHE_DIR` to speed up the bootstrap process. Otherwise you may waste hours on indexing:
    ```bash
    cd $RAG42_CACHE_DIR
    wget https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG/releases/download/BM25Cache/cache.zip
    unzip cache.zip
    ```
    Then start the flask server:
    ```bash
    python server.py # flask server run on localhost:5000
    ```

2. Prepare the frontend.
    ```bash
    cd frontend
    npm install
    npm start # react server run on localhost:3000
    ```

## FAQ
### What LLM models are used in this project?
 - Qwen2.5-0.5B-Instruct
 - Qwen2.5-1.5B-Instruct
 - Qwen2.5-3B-Instruct
 - Qwen2.5-7B-Instruct

You can configure it in `frontend/config.ts` to try more models. This models are used via OpenAI AI. You need to configure `RAG42_OPENAI_API_KEY` and `RAG42_OPENAI_API_PREFIX` corresponding to your LLM service provider.

If you don't have a key, you can also try `Qwen2.5-0.5B-Instruct` locally(via downloading this opensource model from HuggingFace). We support both local HuggingFaceGenerator and OpenAIGenerator to make sure this project can run on laptop.

### How much time/hardware resources are need to start the project.
If you want to build it using docker from scratch, you many need 1 hour or more. The images should take space for about 20GB.

When the server start for the first time, it need to download the HotpotQA dataset and `Qwen2.5-0.5B-Instruct` model from HuggingFace. It will take about 10 minutes and requires additional 1GB.

If you didn't download the cached index file, the bootstrap process of server may take over 3 hours to do the BM25 indexing.

The project is developed and tested on the *Xiaomi Laptop 2016* with:
 - 16GB RAM
 - NVDIA GeForce MX150 GPU
 - Intel i7-8550
 - OS: Arch Linux + KDE