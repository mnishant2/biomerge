#!/bin/bash
# Script to run multiple experiments in parallel on different GPUs

set -e  # Exit on any error

# Make script executable if it's not already
chmod +x scripts/run_gpu.sh

# Function to run experiment in background
run_experiment() {
    local gpu=$1
    local config=$2
    local name=$3
    local extra="${4:-}"
    
    echo "Starting experiment on GPU $gpu: $config ($name)"
    scripts/run_gpu.sh --gpu $gpu --config $config --wandb_name "$name" --extra_args "$extra" &
    
    # Store the PID for later reference
    echo "Started process with PID $! on GPU $gpu"
}

# Example usage - modify these to run your specific experiments
echo "Starting parallel experiments..."

# Setup experiments - modify these to match your needs
# Format: run_experiment GPU_ID CONFIG_NAME WANDB_RUN_NAME "EXTRA_ARGS"

# Run LoRA NER on GPU 0
run_experiment 0 lora_ner "lora_ner_test_run" ""

# Run LoRA EL on GPU 1
run_experiment 1 lora_el "lora_el_test_run" ""

# You can add more experiments if needed (they'll queue up)
# run_experiment 0 fusion "fusion_test_run" "freeze_base_model=true"

echo "All experiments started in parallel. Use 'nvidia-smi' to monitor GPU usage."
echo "To see the logs for each process, check the experiment output directories."
echo ""
echo "Use the following to see all running Python processes:"
echo "ps aux | grep python" 