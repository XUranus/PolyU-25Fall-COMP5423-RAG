#!/bin/bash

# Load environment variables
export $(grep -v '^#' .env | xargs)

echo "Starting application..."
docker-compose up --build