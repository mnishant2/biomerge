#!/bin/bash
#SBATCH --job-name=lora_ner_sweep      # Job name
#SBATCH --partition=gpu                # Partition name (adjust if needed)
#SBATCH --gres=gpu:a100:1              # Request 1 A100 GPU (change :1 to :2 for two)
#SBATCH --time=12:00:00                # Max wall time for the agent (adjust D-HH:MM:SS)
#SBATCH --cpus-per-task=16             # Number of CPU cores per task
#SBATCH --mem=96G                      # Memory per node
#SBATCH --output=logs/%x-%j.out        # Standard output and error log (%x=jobname, %j=jobid)

# --- Debugging and Environment ---
set -e # Exit immediately if a command exits with a non-zero status.
set -x # Print commands and their arguments as they are executed.

echo "Starting sweep agent script execution..."
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
export WANDB_MODE="online"
export WANDB_ENTITY="YOUR_WANDB_ENTITY" # <<< IMPORTANT: Replace with your WandB username or team name
export SWEEP_ID="YOUR_NER_SWEEP_ID"     # <<< IMPORTANT: Replace with the actual Sweep ID from WandB UI

# --- Data Staging (Input) ---
echo "Creating scratch directories..."
mkdir -p "$SCRATCH_DATA_ROOT/union"
mkdir -p "$SCRATCH_OUTPUT_ROOT"

echo "Copying data from $PERSISTENT_DATA_ROOT to $SCRATCH_DATA_ROOT/union ..."
rsync -avzh --no-o --no-g "$PERSISTENT_DATA_ROOT/" "$SCRATCH_DATA_ROOT/union/" # Copy contents into union subdir
echo "Data staging complete."
ls -R $SCRATCH_DATA_ROOT # List contents for verification

# --- Run WandB Agent ---
# Navigate to the code directory
cd "$WORKSPACE_ROOT"

echo "Starting WandB agent for sweep: $WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"

# CRITICAL NOTE FOR SWEEPS WITH SCRATCH:
# The 'command' section of your sweep configuration YAML (used when creating the sweep on WandB)
# MUST include the overrides for data_root and output_root pointing to the scratch directories.
# Example sweep command template (modify as needed):
# command:
#   - ${env}
#   - python
#   - src/main.py
#   - ${args_no_hyphens}         # Passes hyperparameters from sweep
#   - hydra.job.config_name=lora_ner # Base config for the sweep
#   - data_root=$TMPDIR/data     # Pass scratch data path to the script
#   - output_root=$TMPDIR/outputs # Pass scratch output path to the script
#   - +hydra.job.chdir=True       # Ensure Hydra runs from the output dir

# The agent will execute the command defined in the sweep config.
# It will run 'count' jobs (or run indefinitely if count is not set).
wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"

echo "WandB agent finished or exited."

# --- Data Staging (Output) ---
echo "Copying results from $SCRATCH_OUTPUT_ROOT to $ARCHIVE_ROOT ..."
# Ensure archive directory exists
mkdir -p "$ARCHIVE_ROOT"
# Copy the entire output structure generated in scratch by potentially multiple agent runs
rsync -avzh --no-o --no-g "$SCRATCH_OUTPUT_ROOT/" "$ARCHIVE_ROOT/"
echo "Result staging complete."

# --- Cleanup (Optional) ---
# echo "Cleaning up scratch directory $TMPDIR ..."
# rm -rf "$TMPDIR/*"
# echo "Scratch cleanup complete."

echo "Job finished successfully." 