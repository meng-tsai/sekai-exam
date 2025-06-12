#!/bin/bash

# Sekai Optimizer - Docker Run Script

echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo "Running data synthesis..."
python src/sekai_optimizer/scripts/synthesize_data.py

echo "Building search index..."
python src/sekai_optimizer/scripts/build_index.py

echo "Building Sekai Optimizer Docker image..."
docker build -t sekai-optimizer .

echo "Running Sekai Optimizer..."
docker run --rm \
    --env-file .env \
    -v "$(pwd)/src/sekai_optimizer/data:/app/src/sekai_optimizer/data" \
    sekai-optimizer

echo "Sekai Optimizer execution completed." 