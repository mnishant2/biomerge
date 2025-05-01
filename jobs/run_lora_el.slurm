#!/bin/bash
#SBATCH --job-name=lora_el_single      # Job name
#SBATCH --partition=gpu                # Partition name (adjust if needed)
#SBATCH --gres=gpu:a100:1              # Request 1 A100 GPU (change :1 to :2 for two)
#SBATCH --time=04:00:00                # Max wall time (adjust D-HH:MM:SS)
#SBATCH --cpus-per-task=16             # Number of CPU cores per task
#SBATCH --mem=96G                      # Memory per node
#SBATCH --output=logs/%x-%j.out        # Standard output and error log (%x=jobname, %j=jobid)

# --- Debugging and Environment ---
set -e # Exit immediately if a command exits with a non-zero status.
set -x # Print commands and their arguments as they are executed.

echo "Starting job script execution..."
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_TMPDIR: $TMPDIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# --- Setup ---
module purge
module load 2024                   # Adjust module loads as per your HPC environment
module load CUDA/12.2.0            # Match CUDA version if needed
module load Python/3.11.4-GCCcore-13.2.0 # Match Python version

# Activate your Python environment
source "$HOME/envs/taskmerge/bin/activate"
echo "Python environment activated: $(which python)"

# Define base directories
export WORKSPACE_ROOT="/home/nmishra/data/storage_hpc_nishant/biomerge"
export PERSISTENT_DATA_ROOT="$WORKSPACE_ROOT/data/union"
export SCRATCH_DATA_ROOT="$TMPDIR/data"
export SCRATCH_OUTPUT_ROOT="$TMPDIR/outputs"
export ARCHIVE_ROOT="$WORKSPACE_ROOT/archive"

# --- WandB Setup ---
export WANDB_PROJECT="TaskMerge"
export WANDB_MODE="online" # Set to "offline" or "disabled" if needed
# WandB run name will be taken from the Hydra config (e.g., lora_el.yaml -> wandb_name)

# --- Data Staging (Input) ---
echo "Creating scratch directories..."
mkdir -p "$SCRATCH_DATA_ROOT/union"
mkdir -p "$SCRATCH_OUTPUT_ROOT"

echo "Copying data from $PERSISTENT_DATA_ROOT to $SCRATCH_DATA_ROOT/union ..."
rsync -avzh --no-o --no-g "$PERSISTENT_DATA_ROOT/" "$SCRATCH_DATA_ROOT/union/" # Copy contents into union subdir
echo "Data staging complete."
ls -R $SCRATCH_DATA_ROOT # List contents for verification

# --- Run Training ---
# Navigate to the code directory
cd "$WORKSPACE_ROOT"

echo "Starting model training..."
# Notes:
# - `hydra.job.config_name=lora_el`: Specifies the config file (without .yaml) relative to configs/
# - `data_root=$SCRATCH_DATA_ROOT`: Overrides the data root path in the config
# - `output_root=$SCRATCH_OUTPUT_ROOT`: Overrides the output root path in the config
# - Trainer should automatically use available GPUs (`CUDA_VISIBLE_DEVICES`)
srun python src/main.py \
    hydra.job.config_name=lora_el \
    data_root="$SCRATCH_DATA_ROOT" \
    output_root="$SCRATCH_OUTPUT_ROOT" \
    +hydra.job.chdir=True # Ensure Hydra runs from the specified output dir (within scratch)
    # Add any other specific overrides needed for this run below
    # e.g., training.per_device_train_batch_size=X training.gradient_accumulation_steps=Y

echo "Training script finished."

# --- Data Staging (Output) ---
echo "Copying results from $SCRATCH_OUTPUT_ROOT to $ARCHIVE_ROOT ..."
# Ensure archive directory exists
mkdir -p "$ARCHIVE_ROOT"
# Copy the entire output structure generated in scratch
rsync -avzh --no-o --no-g "$SCRATCH_OUTPUT_ROOT/" "$ARCHIVE_ROOT/"
echo "Result staging complete."

# --- Cleanup (Optional) ---
# echo "Cleaning up scratch directory $TMPDIR ..."
# rm -rf "$TMPDIR/*"
# echo "Scratch cleanup complete."

echo "Job finished successfully." 