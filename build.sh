#!/bin/bash
# This script builds the Docker containers for the B-S Architecture RAG42 project.

# Show the configurations in .env
echo "===== Building The RAG42 Project ======"
export $(grep -v '^#' .env | xargs)
echo "RAG42_FRONTEND_PORT = $RAG42_FRONTEND_PORT"
echo "RAG42_BACKEND_PORT = $RAG42_BACKEND_PORT"
echo "RAG42_BACKEND_HOST = $RAG42_BACKEND_HOST"
echo "RAG42_STORAGE_DIR = $RAG42_STORAGE_DIR"
echo "RAG42_CACHE_DIR = $RAG42_CACHE_DIR"
echo "RAG42_OPENAI_API_KEY = $RAG42_OPENAI_API_KEY"
echo "RAG42_OPENAI_API_URL = $RAG42_OPENAI_API_URL"
echo "======================================="


# Create necessary directories
mkdir -p volumes/storage volumes/cache
cwd=$(pwd)
cd volumes/cache
wget https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG/releases/download/BM25Cache/cache.zip
unzip cache.zip
cd $cwd

# Build the containers
echo "Building containers..."
docker-compose build --build-arg RAG42_FRONTEND_PORT=$RAG42_FRONTEND_PORT --build-arg RAG42_BACKEND_PORT=$RAG42_BACKEND_PORT

if [ $? -eq  0 ] ; then
    echo "Build completed successfully!"
    echo "To start the application, run: docker-compose up"
else
    echo "build failed."
fi