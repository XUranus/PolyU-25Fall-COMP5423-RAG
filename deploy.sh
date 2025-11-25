#!/bin/bash
# This script deploys the Docker containers for the RAG42 project using docker-compose.

echo "Starting deployment..."

# Check if containers are already running
if [ "$(docker-compose ps -q)" ]; then
    echo "Stopping existing containers..."
    docker-compose down
fi

# Start the services
echo "Starting services..."
docker-compose up -d

echo "Services started successfully!"
echo "Frontend available at: http://localhost:3000"
echo "Backend available at: http://localhost:5000"

# Optional: Open browser (uncomment if needed)
# if command -v xdg-open &> /dev/null; then
#     xdg-open http://localhost:3000
# elif command -v open &> /dev/null; then
#     open http://localhost:3000
# fi