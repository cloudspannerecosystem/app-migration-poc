#!/bin/bash

# Check for at least 3 arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <source_directory> <access_key> <output_file> [<gemini_version>]"
    exit 1
fi

# Assign arguments to variables
SOURCE_DIRECTORY=$1
ACCESS_KEY=$2
OUTPUT_FILE=$3
GEMINI_VERSION=${4:-default_version}  # Use default_version if the fourth argument is not provided

# Path to the virtual environment
VENV_PATH="/usr/local/google/home/gauravpurohit/ai/app-migration-poc/.venv"

# Path to the Python script
PYTHON_SCRIPT="/usr/local/google/home/gauravpurohit/ai/app-migration-poc/run_script.py"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Invoke the Python script with the arguments
python "$PYTHON_SCRIPT" "$SOURCE_DIRECTORY" "$ACCESS_KEY" "$OUTPUT_FILE" "$GEMINI_VERSION"

# Deactivate the virtual environment
deactivate
