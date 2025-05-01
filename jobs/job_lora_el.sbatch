#!/bin/bash
#SBATCH --job-name=el_lora_single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge
module load 2024
module load CUDA/12.2.0
module load Python/3.11.4-GCCcore-13.2.0

# (OPTIONAL) prevent NCCL IB warnings on Snellius
export NCCL_IB_DISABLE=1

# Activate venv built under the same toolchain
source "$HOME/envs/taskmerge/bin/activate"
echo "Python: $(which python)"

###################  WANDB  ###################################################
# Either rely on ~/.netrc  OR export key here
# export WANDB_API_KEY=$(cat $HOME/.wandb_key)
export WANDB_PROJECT=TaskMerge

###################  PATHS  ###################################################
WORKROOT="$HOME/data/storage_hpc_nishant/biomerge"   # repo root
PERSIST_DATA="$WORKROOT/data/union"
TMP_DATA="$TMPDIR/data"    # fast scratch
TMP_OUT="$TMPDIR/outputs"
mkdir -p "$TMP_DATA/union" "$TMP_OUT" # Ensure union subdir exists

rsync -a --no-o --no-g "$PERSIST_DATA/" "$TMP_DATA/union/" # Copy into union subdir
echo "Data staged to $TMP_DATA"

###################  LAUNCH  ##################################################
cd "$WORKROOT"

echo "Launching EL training with recommended settings for A100..."
# Note: Ensure flash-attn is installed in the environment: pip install flash-attn==2.*

srun python -m src.main \ # Use python directly, srun handles resource allocation
    hydra.job.config_name=lora_el                               \ # Base config for EL
    data_root=$TMP_DATA                                         \ # Use scratch data
    output_root=$TMP_OUT                                        \ # Use scratch output
    +hydra.job.chdir=True                                       \
    model.params.quantization=bf16                              \ # Recommended for A100 speed
    model.params.lora_r=16 model.params.lora_alpha=32           \ # Example EL LoRA params (adjust if needed)
    training.per_device_train_batch_size=2                      \ # Batch size suitable for bf16 on 48GB
    training.gradient_accumulation_steps=16                     \ # Keep effective BS=32 (adjust if bs changes)
    training.dataloader_num_workers=8                           \ # Increase data loader workers
    training.gradient_checkpointing=false                       # Keep disabled if memory allows (faster)
    # training.gradient_checkpointing=true                      # Enable if OOM with bf16

echo "Training completed"

###################  ARCHIVE RESULTS  #########################################
mkdir -p "$WORKROOT/archive"
rsync -a --no-o --no-g "$TMP_OUT/" "$WORKROOT/archive/"
echo "Results archived"
