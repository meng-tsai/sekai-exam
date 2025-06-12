#!/bin/bash

# Sekai Optimizer - Docker Run Script

echo "Building Sekai Optimizer Docker image..."
docker build -t sekai-optimizer .

echo "Running Sekai Optimizer..."
docker run --rm \
    --env-file .env \
    -v "$(pwd)/src/sekai_optimizer/data:/app/src/sekai_optimizer/data" \
    sekai-optimizer

echo "Sekai Optimizer execution completed." 