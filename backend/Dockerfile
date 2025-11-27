# backend dockerfilequit
FROM continuumio/miniconda3:latest

ARG RAG42_BACKEND_PORT

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy conda environment file, requirements
COPY environment.yml ./
COPY requirements.txt ./
COPY db_init.sql ./
COPY *.py ./

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "COMP5423-RAG42", "/bin/bash", "-c"]

# Install Python packages
RUN pip install -r requirements.txt

# Expose port

RUN echo "backend will use port: $RAG42_BACKEND_PORT"
EXPOSE $RAG42_BACKEND_PORT

# Start the Flask application
CMD ["conda", "run", "-n", "COMP5423-RAG42", "python", "server.py"]