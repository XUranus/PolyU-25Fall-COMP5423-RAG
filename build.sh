#!/bin/bash
# This script builds the Docker containers for the B-S Architecture RAG42 project.

echo "Building The RAG42 Project..."

# Create necessary directories
mkdir -p volumes/storage volumes/logs

# Build the containers
echo "Building containers..."
docker-compose build

echo "Build completed successfully!"
echo "To start the application, run: docker-compose up"