#!/bin/bash
# Script to test run an individual model on a specific GPU

set -e  # Exit on any error

# Default values
GPU=0
MODEL_TYPE="ner"  # ner, el, fusion, merge
TEST_BATCH_SIZE=4
WANDB_NAME=""
MODEL_PATH=""
SPLIT="dev"  # Use dev split by default for evaluation

function print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --gpu VALUE        GPU ID to use (default: 0)"
    echo "  --model_type TYPE  Model type to test (ner, el, fusion, merge) (default: ner)"
    echo "  --model_path PATH  Path to model checkpoint (if not provided, will use default)"
    echo "  --batch_size VALUE Test batch size (default: 4)"
    echo "  --wandb_name NAME  Custom name for wandb run (default: auto-generated)"
    echo "  --split SPLIT      Data split to use (dev or test) (default: dev)"
    echo "  --help             Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch_size)
            TEST_BATCH_SIZE="$2"
            shift 2
            ;;
        --wandb_name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
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

# Validate split
if [[ "$SPLIT" != "dev" && "$SPLIT" != "test" ]]; then
    echo "Invalid split: $SPLIT (must be 'dev' or 'test')"
    exit 1
fi

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$GPU
echo "Testing on GPU: $GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

# Set config based on model type
case $MODEL_TYPE in
    ner)
        CONFIG="lora_ner"
        ;;
    el)
        CONFIG="lora_el"
        ;;
    fusion)
        CONFIG="fusion"
        ;;
    merge)
        CONFIG="merge"
        ;;
    *)
        echo "Unknown model type: $MODEL_TYPE"
        print_usage
        exit 1
        ;;
esac

# Set wandb name if provided
if [ -n "$WANDB_NAME" ]; then
    WANDB_ARGS="wandb_name=$WANDB_NAME"
else
    WANDB_ARGS="wandb_name=${MODEL_TYPE}_${SPLIT}_gpu${GPU}"
fi

# Set model path if provided
if [ -n "$MODEL_PATH" ]; then
    MODEL_ARGS="model.params.path=$MODEL_PATH"
else
    MODEL_ARGS=""
fi

# Run the evaluation command
echo "Testing $MODEL_TYPE model on $SPLIT split with config: $CONFIG"
echo "Wandb run name: ${WANDB_NAME:-auto-generated}"

# Ensure max_length is set to 256 for consistency
EXTRA_ARGS="datamodule.params.max_length=256 datamodule.params.splits=[$SPLIT] eval_split=$SPLIT"

CMD="python src/main.py --config-name=$CONFIG mode=eval test_batch_size=$TEST_BATCH_SIZE $WANDB_ARGS $MODEL_ARGS $EXTRA_ARGS"
echo "Running command: $CMD"
eval $CMD

echo "Testing complete!" 