#!/usr/bin/env python
import os
import sys
import hydra
import logging
import torch
import numpy as np
import re
import json
import wandb
from omegaconf import OmegaConf, DictConfig
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from peft import PeftModel, PeftConfig
import importlib

# Import necessary callback functions
from src.utils.callbacks import get_callbacks_for_task

logger = logging.getLogger(__name__)

def evaluate_model(cfg: DictConfig):
    """Evaluate a model using Trainer and task-specific callbacks."""
    logger.info(f"Evaluating model with configuration: {OmegaConf.to_yaml(cfg)}")

    # --- Setup --- 
    eval_split = cfg.get("eval_split", "dev")
    model_type = cfg.get("model_type", cfg.get("task", "unknown"))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_merged_or_fused = model_type in ["fusion", "merge"]
    
    # Determine tasks to evaluate based on model type
    tasks_to_evaluate = []
    if is_merged_or_fused:
        logger.info(f"Evaluating {model_type} model on both NER and EL tasks.")
        tasks_to_evaluate = ["ner", "el"]
    elif model_type in ["ner", "el", "joint"]:
        tasks_to_evaluate = [model_type]
    else:
        raise ValueError(f"Unsupported model_type for evaluation: {model_type}")

    # --- Load Model and Base Tokenizer --- 
    logger.info("Loading base model and tokenizer")
    model_fn_mod, model_fn_name = cfg.model._target_.rsplit(".", 1)
    build_model = getattr(importlib.import_module(model_fn_mod), model_fn_name)
    model = build_model(**cfg.model.params)
    tokenizer = AutoTokenizer.from_pretrained(cfg.datamodule.params.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set tokenizer pad_token to eos_token")

    all_results = {}
    
    # --- Iterate through tasks for evaluation --- 
    for task in tasks_to_evaluate:
        logger.info(f"--- Evaluating on task: {task} (split: {eval_split}) --- ")
        task_output_dir = output_dir / task # Subdirectory for each task
        task_output_dir.mkdir(exist_ok=True)
        
        # Determine datamodule target for the task
        if task == "ner":
            dm_target = cfg.datamodule.get("ner_target", "src.datamodules.ner_module.NERDataModule")
            dm_params = cfg.datamodule.get("ner_params", cfg.datamodule.params)
        elif task == "el":
            dm_target = cfg.datamodule.get("el_target", "src.datamodules.el_module.ELDataModule")
            dm_params = cfg.datamodule.get("el_params", cfg.datamodule.params)
        elif task == "joint":
             dm_target = cfg.datamodule.get("joint_target", "src.datamodules.joint_decode_module.JointDecodeModule")
             dm_params = cfg.datamodule.get("joint_params", cfg.datamodule.params)
        else:
             logger.error(f"Cannot determine datamodule for unknown task: {task}")
             continue
        
        # Load data for the specific task and split
        logger.info(f"Initializing datamodule for task '{task}'")
        dm_module, dm_class = dm_target.rsplit(".", 1)
        DataModule = getattr(importlib.import_module(dm_module), dm_class)
        # Only load the eval_split
        datamodule = DataModule(splits=[eval_split], tokenizer_name=cfg.datamodule.params.tokenizer_name, **dm_params)
        
        eval_dataset = datamodule.datasets.get(eval_split)
        if not eval_dataset:
            logger.error(f"Could not load {eval_split} dataset for task {task}")
            continue
        logger.info(f"Loaded {len(eval_dataset)} examples for task '{task}'")

        # --- Setup Trainer for Evaluation --- 
        # Callbacks will handle metric calculation
        task_callbacks = get_callbacks_for_task(task, tokenizer)
        
        # Determine if generation is needed for the current task
        use_generate = task in ["el", "joint"]
        logger.info(f"Setting predict_with_generate={use_generate} for task '{task}'")

        # Use TrainingArguments for evaluation settings
        eval_args = TrainingArguments(
            output_dir=str(task_output_dir),
            per_device_eval_batch_size=cfg.get("test_batch_size", 8),
            dataloader_num_workers=cfg.get("num_workers", 2),
            remove_unused_columns=False, # Keep original columns
            report_to=[], # Callbacks handle WandB logging
            fp16=cfg.get("fp16", torch.cuda.is_available()),
            # Important: Set predict_with_generate based on the task
            predict_with_generate=use_generate,
            # generation_config=model.generation_config # Use model's default or specify
            # generation_max_length=cfg.get("max_length", 256) # Ensure generation length is reasonable
        )
        
        trainer = Trainer(
            model=model,
            args=eval_args,
            tokenizer=tokenizer,
            callbacks=task_callbacks,
            # No compute_metrics needed here as callbacks handle it
        )

        # --- Run Evaluation --- 
        logger.info(f"Running Trainer.evaluate for task '{task}'...")
        # The callbacks attached to the Trainer will compute and log metrics during this call
        eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
        
        # Metrics are automatically populated by the callbacks
        logger.info(f"Metrics computed by callbacks for task '{task}': {json.dumps(eval_metrics, indent=2)}")
        
        # Store results, prefixing with task name
        for key, value in eval_metrics.items():
             # Standard trainer metrics start with 'eval_', callback metrics might not
             metric_key = key if key.startswith('eval_') else f"eval_{key}" # Ensure 'eval_' prefix
             # Add task prefix (e.g., ner_eval_f1)
             all_results[f"{task}_{metric_key}"] = value 

    # --- Final Logging --- 
    logger.info(f"--- Overall Evaluation Completed --- ")
    logger.info(json.dumps(all_results, indent=2))
    
    # Log combined results to a single WandB run if desired and not disabled
    if not cfg.get("disable_wandb", False):
        # Ensure wandb is initialized (might be done by Trainer callbacks, but check)
        wandb_project = cfg.get("wandb_project", "biomedical-entity-model-evaluation")
        # Use a name reflecting the overall evaluation
        wandb_name = cfg.get("wandb_name", f"eval_{model_type}_{eval_split}")
        if wandb.run is None:
             wandb.init(project=wandb_project, name=wandb_name, config=OmegaConf.to_container(cfg, resolve=True))
        
        # Log all collected metrics, prefixed with the split
        wandb_metrics = {f"{eval_split}/{k}": v for k, v in all_results.items()} 
        wandb.log(wandb_metrics)
        logger.info(f"Logged combined metrics to WandB run: {wandb.run.name}")
        # Let main script or outer context finish wandb run
        
    # Save combined metrics to file
    metrics_file = output_dir / f"evaluation_{eval_split}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Saved combined metrics to {metrics_file}")

    return all_results

# Ensure old/obsolete functions are removed or commented out if necessary
# def evaluate_ner(...)
# def evaluate_el(...)
# def evaluate_joint(...)
# def evaluate_merged_model(...)

# Main entry point for standalone evaluation
@hydra.main(config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    logger.info(f"Running evaluation with config: \n{OmegaConf.to_yaml(cfg)}")
    evaluate_model(cfg) # Call the unified evaluation function

if __name__ == "__main__":
    main() 