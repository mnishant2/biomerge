#!/bin/bash
# Run all the smaller model training jobs interactively on A10 GPUs

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1  # Use both A10 GPUs

# Create output directories
mkdir -p outputs/lora_ner
mkdir -p outputs/lora_el
mkdir -p outputs/sapbert
mkdir -p logs

# Train LoRA NER model
echo "===== Training LoRA NER Model ====="
python src/train.py configs/lora_ner.yaml 2>&1 | tee logs/lora_ner.log

# Train LoRA EL model
echo "===== Training LoRA EL Model ====="
python src/train.py configs/lora_el.yaml 2>&1 | tee logs/lora_el.log

# interactive A10
python -m src.train configs/biobert.yaml

# evaluation on test set

# Train SapBERT model
echo "===== Training SAPBERT Model ====="
python src/train.py configs/sapbert.yaml 2>&1 | tee logs/sapbert.log

# Build FAISS index after SapBERT training
echo "===== Building FAISS Index from SapBERT Embeddings ====="
python scripts/build_indices.py \
    --model-path outputs/sapbert \
    --concept-file data/union/SAPBERT/concepts.jsonl \
    --output-dir data/indices \
    2>&1 | tee logs/build_indices.log

# Evaluate merged NER+EL adapters
echo "===== Evaluating Merged Adapters ====="
python src/evaluate.py \
    model_class._target_=src.models.full_llama.FullLlamaModel \
    model_path=outputs/lora_ner/merged \
    tokenizer_path=meta-llama/Llama-3-8B \
    test_path=data/union/joint_decode_EL/union_test_joint_decode.jsonl \
    task=joint \
    output_dir=outputs/merged_eval \
    2>&1 | tee logs/merged_eval.log
python -m src.evaluate \
        --checkpoint outputs/biobert_ner \
        --task biobert \
        --split test

echo "===== All training jobs completed! =====" 