#!/bin/bash
#SBATCH --job-name=llama_fullft
#SBATCH --output=logs/llama_fullft_%j.out
#SBATCH --error=logs/llama_fullft_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpu

# Load modules
module purge
module load python/3.10
module load cuda/12.1

# Activate virtual environment
source /path/to/venv/bin/activate

# Set environment variables for distributed training
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12802
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training with deepspeed
python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/train.py \
    configs/full_llama.yaml \
    --deepspeed=configs/deepspeed_zero3.json 