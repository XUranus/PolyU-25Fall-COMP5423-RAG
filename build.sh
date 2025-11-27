#!/bin/bash
# This script builds the Docker containers for the B-S Architecture RAG42 project.

echo "Building The RAG42 Project..."

# Create necessary directories
mkdir -p volumes/storage volumes/cache
cwd=$(pwd)
cd volumes/cache
wget https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG/archive/refs/tags/BM25Cache.zip
unzip BM25Cache.zip
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