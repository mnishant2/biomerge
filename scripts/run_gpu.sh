#!/bin/bash
# Script to run experiments on specific GPUs with custom wandb names

set -e  # Exit on any error

# Default values
GPU=0
CONFIG="defaults"
WANDB_NAME=""
EXTRA_ARGS=""
# MAX_LENGTH=256 # No longer needed as script argument

function print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --gpu VALUE        GPU ID to use (default: 0)"
    echo "  --config NAME      Configuration file to use (default: defaults)"
    echo "  --wandb_name NAME  Custom name for wandb run (default: auto-generated)"
    # echo "  --max_length VALUE Maximum sequence length to use (default: 256)" # Removed
    echo "  --extra_args ARGS  Extra arguments to pass to the Python script (use quotes)"
    echo "  --help             Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --wandb_name)
            WANDB_NAME="$2"
            shift 2
            ;;
        # --max_length)
        #     MAX_LENGTH="$2"
        #     shift 2
        #     ;;
        --extra_args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$GPU
echo "Running on GPU: $GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

# Silence tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# Set wandb name if provided
if [ -n "$WANDB_NAME" ]; then
    WANDB_ARG="wandb_name=$WANDB_NAME"
else
    WANDB_ARG=""
fi

# Remove the max_length override as it's defined in config files
# LENGTH_ARG="+datamodule.params.max_length=$MAX_LENGTH"

# Run the Python command as a module using -m src.main
echo "Starting experiment with config: $CONFIG"
# echo "Max sequence length: $MAX_LENGTH" # Removed
echo "Wandb run name: ${WANDB_NAME:-auto-generated}"

# Use python3 from the conda environment if needed (adjust path if necessary)
PYTHON_CMD="python3"
if command -v conda &> /dev/null && conda list | grep -q biomerge;
 then
    PYTHON_CMD="conda run -n biomerge python3"
    echo "Using Conda environment: biomerge"
fi

CMD="$PYTHON_CMD -m src.main --config-name=$CONFIG $WANDB_ARG $EXTRA_ARGS"
echo "Running command: $CMD"
eval $CMD 