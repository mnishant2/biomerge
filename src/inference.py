#!/usr/bin/env python
import os
import sys
import hydra
import logging
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Load input data (file or single text)
    if cfg.input_file:
        if cfg.input_file.endswith('.jsonl'):
            input_data = [json.loads(line) for line in open(cfg.input_file, 'r')]
        else:
            # Assume text file with one input per line
            input_data = [{"input": line.strip()} for line in open(cfg.input_file, 'r')]
    elif cfg.input_text:
        input_data = [{"input": cfg.input_text}]
    else:
        logger.error("No input provided. Set either input_file or input_text in config.")
        return
    
    # Load model and tokenizer
    model_cls = hydra.utils.instantiate(cfg.model_class)
    model = model_cls.from_pretrained(cfg.model_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path or cfg.model_path)
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process based on task
    results = []
    for item in tqdm(input_data):
        if cfg.task == "ner":
            result = process_ner(model, tokenizer, item, cfg)
        elif cfg.task == "el":
            result = process_el(model, tokenizer, item, cfg)
        elif cfg.task == "joint":
            result = process_joint(model, tokenizer, item, cfg)
        else:
            logger.error(f"Unknown task: {cfg.task}")
            return
        
        results.append(result)
    
    # Save results
    output_file = cfg.output_file or "inference_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def process_ner(model, tokenizer, item, cfg):
    # Process NER using token classification
    inputs = tokenizer(item["input"], return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted labels
    predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Convert predictions to labels and align with tokens
    labeled_tokens = []
    for token, pred_id in zip(tokens, predictions):
        label = model.config.id2label.get(pred_id, "O")
        if not token.startswith("##"):  # Basic handling for BERT-style tokenization
            labeled_tokens.append((token, label))
    
    return {
        "input": item["input"],
        "tokens": labeled_tokens
    }


def process_el(model, tokenizer, item, cfg):
    # Process EL using generative approach
    inputs = tokenizer(item["input"], return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    # Extract entity links (e.g., from markdown format)
    entity_links = re.findall(r'\[([^]]+)\]\(([^)]+)\)', generated_text)
    
    return {
        "input": item["input"],
        "generated_text": generated_text,
        "entity_links": entity_links
    }


def process_joint(model, tokenizer, item, cfg):
    # Process joint NER+EL
    inputs = tokenizer(item["input"], return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    # For joint models, the output could be in various formats
    # Here we assume markdown format as seen in the data sample
    
    return {
        "input": item["input"],
        "generated_text": generated_text
    }


if __name__ == "__main__":
    main() 