#!/bin/bash
set -e

# Load environment variables from .env file
set -a
source .env
set +a

echo "Using service account authentication from .env file..."

python3 process_images.py

python3 googlevision_translater.py

python3 note_translater.py

python3 export_responses.py