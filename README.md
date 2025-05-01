# Biomedical Entity Model Merging

## Overview
This project explores training specialized biomedical entity extraction models and merging them efficiently. The project focuses on two key tasks:
1. **Named Entity Recognition (NER)**: Identifying biomedical entity mentions in text
2. **Entity Linking (EL)**: Linking entity mentions to unique concept IDs (CUIs)

The project investigates parameter-efficient fine-tuning, model merging, and adapter fusion approaches to combine task-specific capabilities into a single model.

## Project Structure
- `configs/`: Configuration files for training, evaluation, and model merging
- `src/`: Source code for models, data modules, and utilities
- `scripts/`: Scripts for running experiments and sweeps
- `data/`: Datasets for NER and EL tasks

## Experiments
This project implements the following experiments:

1. **NER-LoRA**: Fine-tune Llama3-OpenBioLLM-8B for the NER task with LoRA
2. **EL-LoRA**: Fine-tune Llama3-OpenBioLLM-8B for generative entity linking
3. **Joint Decode EL**: Train for joint entity recognition and linking using markdown annotations
4. **Model Merging**: Combine LoRA adapters from NER and EL using weight averaging
5. **Adapter Fusion**: Combine adapters with learnable gating mechanism

## Metrics
- **NER**: Span-level precision, recall, F1
- **EL**: Mention-level precision, recall, F1 on (surface, CUI) pairs, sentence-level exact match, unique CUI recall
- **Joint**: Both NER and EL metrics

## Running Experiments

### Quick Start
Use the provided shell script to run experiments:

```bash
# Train NER model
./scripts/run_experiments.sh train-ner

# Train EL model
./scripts/run_experiments.sh train-el

# Train joint model
./scripts/run_experiments.sh train-joint

# Merge NER and EL adapters (with default alpha=0.5)
./scripts/run_experiments.sh merge

# Merge with custom alpha value
./scripts/run_experiments.sh merge --alpha 0.7

# Train adapter fusion
./scripts/run_experiments.sh fusion

# Evaluate models
./scripts/run_experiments.sh eval-ner
./scripts/run_experiments.sh eval-el
./scripts/run_experiments.sh eval-joint

# Evaluate merged model
./scripts/run_experiments.sh eval-merged --model-path outputs/merged_models/ner_el_merged_0.5

# Run hyperparameter sweeps
./scripts/run_experiments.sh sweep-ner
./scripts/run_experiments.sh sweep-el
./scripts/run_experiments.sh sweep-joint
```

### Configuration
All experiments use configuration files in the `configs/` directory. You can customize these to change model parameters, training settings, and evaluation metrics.

### Hyperparameter Sweeps
The project includes W&B sweep configurations for each task:
- `configs/sweep/lora_ner.yaml`: Grid search for NER
- `configs/sweep/lora_el.yaml`: Bayesian optimization for EL
- `configs/sweep/lora_joint.yaml`: Random search for joint model

Each sweep optimizes the following hyperparameters:
- LoRA rank (r): 4, 8, 16
- LoRA dropout: 0.05, 0.1
- Learning rate: 1e-4, 2e-4, 3e-4

## Requirements
See `requirements.txt` for dependencies.

## Project Overview

The goal is to validate whether model merging helps combine model abilities and can be on par with full fine-tuning or multi-task learning approaches, while being faster to build post-hoc.

We implement and compare several approaches:

1. **Task-specific models**:
   - BioBERT for token-wise NER
   - LoRA adapter for Llama-3-8B for NER
   - LoRA adapter for Llama-3-8B for EL
   - SAPBERT embedding-based contrastive learning for EL

2. **Combined models**:
   - Weight-merged NER+EL models with different merging strategies
   - Gated adapter fusion across layers
   - Head concat model (LoRA backbone with dual heads)
   - Full Llama fine-tune on both tasks jointly
   - Joint decode model generating both NER and EL in markdown format

## Directory Structure

```
task-merge/
├── data/
│   ├── union/
│   │   ├── ner/                  ← Token classification data
│   │   ├── el/                   ← Entity linking data
│   │   ├── fullft/               ← Joint task data for full finetune
│   │   ├── head_concat/          ← Data for dual-head model
│   │   ├── SAPBERT/              ← Contrastive learning pairs
│   │   └── joint_decode_EL/      ← Markdown format for joint decoding
│   └── indices/                  ← FAISS & tokenizer caches
├── src/
│   ├── datamodules/
│   │   ├── ner_module.py         ← HuggingFace Dataset + collator
│   │   ├── el_module.py
│   │   ├── headconcat_module.py
│   │   └── sapbert_module.py
│   ├── models/
│   │   ├── lora_ner.py
│   │   ├── lora_el.py
│   │   ├── headconcat.py
│   │   └── full_llama.py
│   ├── train.py                  ← Single entry point for training
│   ├── evaluate.py               ← Evaluation script
│   └── inference.py              ← Inference script
├── configs/
│   ├── lora_ner.yaml
│   ├── lora_el.yaml
│   ├── headconcat.yaml
│   ├── full_llama.yaml
│   └── sapbert.yaml
├── jobs/
│   ├── llama_fullft.sbatch       ← SLURM job for full model training
│   └── sweep_adapterfusion.yaml  ← WandB sweep for adapter fusion
├── scripts/
│   ├── build_indices.py          ← Build FAISS after SapBERT FT
│   └── submit_all_small.sh       ← A10 interactive jobs
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd task-merge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

## Training Models

### Small Models (A10 GPUs)

Run the interactive training script for models that fit on A10 GPUs:

```bash
bash scripts/submit_all_small.sh
```

This will train:
- LoRA adapter for NER
- LoRA adapter for EL
- SAPBERT model

### Large Models (A100 GPUs)

Submit SLURM jobs for full model training:

```bash
sbatch jobs/llama_fullft.sbatch
```

### Adapter Fusion

Run WandB sweeps to find optimal adapter fusion settings:

```bash
wandb sweep jobs/sweep_adapterfusion.yaml
wandb agent <sweep-id>
```

## Evaluation

Evaluate models on test data:

```bash
python src/evaluate.py \
    model_class._target_=src.models.full_llama.FullLlamaModel \
    model_path=<model-path> \
    tokenizer_path=meta-llama/Llama-3-8B \
    test_path=data/union/joint_decode_EL/union_test_joint_decode.jsonl \
    task=joint \
    output_dir=outputs/evaluation
```

## Inference

Run inference on new data:

```bash
python src/inference.py \
    model_class._target_=src.models.full_llama.FullLlamaModel \
    model_path=<model-path> \
    tokenizer_path=meta-llama/Llama-3-8B \
    input_file=<input-file> \
    task=joint \
    output_file=<output-file>
```

## License

[Specify license information]

## Citation

[If applicable] 