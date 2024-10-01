#!/bin/bash

# Check for at least 3 arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <source_directory> <access_key_file> <output_file> [<gemini_version>]"
    exit 1
fi

# Assign arguments to variables
SOURCE_DIRECTORY=$1
MYSQL_SCHEMA_FILE=$2
SPANNER_SCHEMA_FILE=$3
ACCESS_KEY_FILE=$4
OUTPUT_FILE=$5
GEMINI_VERSION=${6:-gemini-1.5-pro-001}

# Path to this shell script
# We assume the script is in the same directory as the venv and our Python command
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Path to the virtual environment
VENV_PATH="$SCRIPT_DIR/.venv"

# Path to the Python script
PYTHON_SCRIPT="$SCRIPT_DIR/run_script.py"

# Read the access key from the file
if [ -f "$ACCESS_KEY_FILE" ]; then
    ACCESS_KEY=$(cat "$ACCESS_KEY_FILE")
else
    echo "Error: Access key file not found: $ACCESS_KEY_FILE"
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Invoke the Python script with the arguments
python "$PYTHON_SCRIPT" "$SOURCE_DIRECTORY" "$MYSQL_SCHEMA_FILE" "$SPANNER_SCHEMA_FILE" "$ACCESS_KEY" "$OUTPUT_FILE" "$GEMINI_VERSION"

# Deactivate the virtual environment
deactivate
