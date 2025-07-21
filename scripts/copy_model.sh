#!/bin/bash
# Script to copy a model to SLURM_TMPDIR at the start of a job

# Usage: ./copy_model.sh model_alias [quantized]
# Example: ./copy_model.sh l_8i  # Copy Llama-3.1-8B-Instruct
# Example: ./copy_model.sh l_8i quantized  # Copy Llama-3.1-8B-Instruct (with flag indicating quantization will be used)

if [ -z "$1" ]; then
    echo "Usage: ./copy_model.sh model_alias [quantized]"
    exit 1
fi

MODEL_ALIAS=$1
USE_QUANTIZED=false

if [ "$2" = "quantized" ]; then
    USE_QUANTIZED=true
    echo "Note: The 'quantized' parameter only indicates that quantization will be used during loading."
fi

# Ensure SLURM_TMPDIR exists
if [ -z "$SLURM_TMPDIR" ]; then
    echo "Error: SLURM_TMPDIR not set. Are you running on a SLURM cluster?"
    exit 1
fi

# Load configuration
CONFIG_FILE="/home/edarsem/projects/def-azouaq/edarsem/soft_knowledge_retrieval/config.ini"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

# Parse config.ini to get model paths
parse_config() {
    local section=$1
    local key=$2
    grep -A100 "^\[$section\]" "$CONFIG_FILE" | grep -m1 "^$key =" | sed 's/^[^=]*= *//'
}

# Get model name from alias
get_model_name() {
    local alias=$1
    parse_config "model_aliases" "$alias"
}

MODEL_NAME=$(get_model_name "$MODEL_ALIAS")
if [ -z "$MODEL_NAME" ]; then
    echo "Error: Model alias '$MODEL_ALIAS' not found in config"
    exit 1
fi

# We're always copying the base model, not a pre-quantized version
# Quantization is applied at loading time, not by using a separate model path
MODEL_PATH=$(parse_config "models" "$MODEL_NAME")
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model '$MODEL_NAME' not found in config"
    exit 1
fi

# Get base model name (last part of the path)
BASE_MODEL_NAME=$(basename "$MODEL_PATH")

# Check model directories in priority order
SOURCE_PATH=""

# 1. First, try the priority directory (project directory - faster but smaller)
MODEL_DIR_PRIORITY=$(parse_config "paths" "models_dir_priority")
if [ -n "$MODEL_DIR_PRIORITY" ] && [ -d "$MODEL_DIR_PRIORITY/$BASE_MODEL_NAME" ]; then
    SOURCE_PATH="$MODEL_DIR_PRIORITY/$BASE_MODEL_NAME"
    echo "Found model in priority directory (faster storage): $SOURCE_PATH"
fi

# 2. If not found in priority directory, try the main models directory (scratch - slower but larger)
if [ -z "$SOURCE_PATH" ]; then
    MODEL_DIR=$(parse_config "paths" "models_dir")
    if [ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR/$BASE_MODEL_NAME" ]; then
        SOURCE_PATH="$MODEL_DIR/$BASE_MODEL_NAME"
        echo "Found model in main directory: $SOURCE_PATH"
    fi
fi

# 3. Try any other configured model directories as fallback
if [ -z "$SOURCE_PATH" ]; then
    for dir_key in models_dir1 models_dir2; do
        MODEL_DIR=$(parse_config "paths" "$dir_key")
        if [ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR/$BASE_MODEL_NAME" ]; then
            SOURCE_PATH="$MODEL_DIR/$BASE_MODEL_NAME"
            echo "Found model in $dir_key: $SOURCE_PATH"
            break
        fi
    done
fi

if [ -z "$SOURCE_PATH" ]; then
    echo "Error: Model not found in any configured model directory"
    exit 1
fi

DEST_PATH="$SLURM_TMPDIR/$BASE_MODEL_NAME"

# Check if model already exists in SLURM_TMPDIR
if [ -d "$DEST_PATH" ]; then
    echo "Model already exists in SLURM_TMPDIR: $DEST_PATH"
else
    echo "Copying model from $SOURCE_PATH to $DEST_PATH..."
    cp -r "$SOURCE_PATH" "$DEST_PATH"
fi

echo "Model is available at: $DEST_PATH"
