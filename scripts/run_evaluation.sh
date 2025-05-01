#!/bin/bash
# Script to evaluate models on NER and EL tasks using both GPUs in parallel

set -e  # Exit on any error

# Make scripts executable
chmod +x scripts/test_model.sh
chmod +x scripts/run_gpu.sh

# Function to run evaluation in the background
run_eval() {
    local gpu=$1
    local model_type=$2
    local split=$3
    local model_path="${4:-}"
    local extra="${5:-}"
    
    if [ -n "$model_path" ]; then
        model_path_arg="--model_path $model_path"
    else
        model_path_arg=""
    fi
    
    echo "Starting evaluation of $model_type on GPU $gpu using $split split"
    scripts/test_model.sh --gpu $gpu --model_type $model_type --split $split $model_path_arg $extra &
    
    # Store the PID for later reference
    echo "Started process with PID $! on GPU $gpu"
}

# Display available options
function print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mode VALUE       Evaluation mode (default: standard)"
    echo "                     Values: standard, fusion, all"
    echo "  --split VALUE      Data split to use (default: dev)"
    echo "                     Values: dev, test"
    echo "  --help             Show this help message"
    echo ""
    echo "Modes:"
    echo "  standard: Evaluate NER and EL models separately"
    echo "  fusion:   Evaluate fusion and merge models"
    echo "  all:      Evaluate all models"
}

# Default values
MODE="standard"
SPLIT="dev"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
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

echo "Starting evaluation in $MODE mode using $SPLIT split"

case $MODE in
    standard)
        # Run NER on GPU 0 and EL on GPU 1
        run_eval 0 ner $SPLIT
        run_eval 1 el $SPLIT
        ;;
    fusion)
        # Run fusion on GPU 0 and merge on GPU 1
        run_eval 0 fusion $SPLIT
        run_eval 1 merge $SPLIT
        ;;
    all)
        # Run all evaluations (sequentially on each GPU)
        run_eval 0 ner $SPLIT
        run_eval 1 el $SPLIT
        
        # Wait for first batch to complete
        wait
        
        # Run fusion and merge
        run_eval 0 fusion $SPLIT
        run_eval 1 merge $SPLIT
        ;;
    *)
        echo "Unknown mode: $MODE"
        print_usage
        exit 1
        ;;
esac

echo "All evaluation processes started. Monitor progress with 'nvidia-smi'."
echo "Use 'ps aux | grep python' to see running processes."
echo "Results will be logged to wandb." 