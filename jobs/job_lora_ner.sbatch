#!/bin/bash
#SBATCH --job-name=ner_lora_single
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

# ---------- Python virtual‑env ----------
source "$HOME/envs/taskmerge/bin/activate"

# ---------- Paths ----------
WORKROOT="$HOME/data/storage_hpc_nishant/biomerge"   # repo root
PERSIST_DATA="$WORKROOT/data/union"
TMP_DATA="$TMPDIR/data"    # fast scratch
TMP_OUT="$TMPDIR/outputs"
mkdir -p "$TMP_DATA/union" "$TMP_OUT" # Ensure union subdir exists

rsync -a --no-o --no-g "$PERSIST_DATA/" "$TMP_DATA/union/" # Copy into union subdir

# ---------- wandb ( ~/.netrc preferred ) ----------
export WANDB_PROJECT=TaskMerge
# export WANDB_API_KEY=$(cat $HOME/.wandb_key)   # optional

cd "$WORKROOT"

echo "Launching training with recommended settings for A100..."
# Note: Ensure flash-attn is installed in the environment: pip install flash-attn==2.*

srun torchrun --nproc_per_node 1 --master_port 29500 \ 
    -m src.main                                                 \
     hydra.job.config_name=lora_ner                              \ # Base config
     data_root=$TMP_DATA                                         \ # Use scratch data
     output_root=$TMP_OUT                                        \ # Use scratch output
     +hydra.job.chdir=True                                       \
     model.params.quantization=bf16                              \ # Recommended for A100 speed
     model.params.r=8 model.params.alpha=16                      \
     training.per_device_train_batch_size=2                      \ # Batch size suitable for bf16 on 48GB
     training.gradient_accumulation_steps=16                     \ # Keep effective BS=32
     training.dataloader_num_workers=8                           \ # Increase data loader workers
     training.gradient_checkpointing=false                       # Keep disabled if memory allows (faster)
     # training.gradient_checkpointing=true                      # Enable if OOM with bf16

# ---------- archive results ----------
rsync -a --no-o --no-g "$TMP_OUT/" "$WORKROOT/archive/"