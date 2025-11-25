# backend dockerfilequit
FROM continuumio/miniconda3:latest

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

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Start the Flask application
CMD ["conda", "run", "-n", "COMP5423-RAG42", "python", "server.py"]