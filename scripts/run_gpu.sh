#!/bin/bash
# Script to run experiments on specific GPUs with custom wandb names

set -e  # Exit on any error

# Default values
GPU=0
CONFIG="defaults"
WANDB_NAME=""
EXTRA_ARGS=""

# --- Default Recommended Performance Settings --- 
# Can be overridden by --extra_args or the specific config file loaded
# Suitable defaults for A100/H100 with 48GB+ VRAM and bf16
DEFAULT_OVERRIDES=(
    "model.params.quantization=bf16"
    "training.dataloader_num_workers=8" # Adjust based on local CPU cores
    "training.gradient_checkpointing=false"
    "training.per_device_train_batch_size=2" # Suitable starting point for bf16
    "training.gradient_accumulation_steps=16" # Keep effective BS=32
)
DEFAULT_OVERRIDES_STR=$(printf " %s" "${DEFAULT_OVERRIDES[@]}")
# Remove leading space
DEFAULT_OVERRIDES_STR=${DEFAULT_OVERRIDES_STR:1}

function print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --gpu VALUE        GPU ID to use (default: 0)"
    echo "  --config NAME      Hydra configuration NAME to use (e.g., lora_ner, defaults)"
    echo "  --wandb_name NAME  Custom name for wandb run (default: taken from config)"
    echo "  --extra_args ARGS  Extra arguments to pass to the Python script (use quotes, will override defaults set in this script)"
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
        --extra_args)
            EXTRA_ARGS="$2" # User args will override defaults
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

# Combine default overrides with user-provided extra args
# User args take precedence if they specify the same keys
ALL_ARGS="$DEFAULT_OVERRIDES_STR $EXTRA_ARGS"

# --- Task-Specific Overrides --- 
# Ensure predict_with_generate=False for NER evaluation during training
if [[ "$CONFIG" == "lora_ner" ]]; then
    echo "Applying task-specific override for NER: training.predict_with_generate=false"
    ALL_ARGS="$ALL_ARGS training.predict_with_generate=false"
fi

# Run the Python command as a module using -m src.main
echo "Starting experiment with config: $CONFIG"
echo "Default Overrides: $DEFAULT_OVERRIDES_STR"
echo "Extra User Args: $EXTRA_ARGS"
echo "Wandb run name: ${WANDB_NAME:-taken from config}"

# Determine Python command (handle conda env)
PYTHON_CMD="python"
if command -v conda &> /dev/null && conda info --envs | grep -q taskmerge;
 then
    # Prefer activating the environment if possible for cleaner setup
    # This might require sourcing conda setup script first, depending on shell init
    # Falling back to `conda run` for robustness
    PYTHON_CMD="conda run --no-capture-output -n taskmerge python"
    echo "Using Conda environment via 'conda run': taskmerge"
else
    echo "Using Python command: $PYTHON_CMD (ensure correct env is active)"
fi

# Construct the final command
# Using --config-name ensures it loads the correct base config from the configs/ directory
CMD="$PYTHON_CMD -m src.main --config-name=$CONFIG $WANDB_ARG $ALL_ARGS"
echo "Running command: $CMD"

eval $CMD

echo "Experiment finished." 