#!/bin/bash

# Load environment variables from .env file
set -a
source .env
set +a

echo "Using service account authentication from .env file..."

python3 process_images.py

python3 googlevision-translater.py

python3 gpt4-note-translater.py

python3 export_responses.py