#!/bin/bash
# Check if a file path argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_json_file>"
    exit 1
fi

# Get the file path from the argument
JSON_FILE_PATH="$1"

poetry run python app/federated_learning_sweep.py --json-path "$JSON_FILE_PATH"

