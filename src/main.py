#!/usr/bin/env python
import os
import sys
import logging
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import importlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    """Main entry point for all tasks: training, evaluation, and inference."""
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Process the command based on mode
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "merge":
        merge_adapters(cfg)
    elif cfg.mode == "fusion":
        adapter_fusion(cfg)
    else:
        logger.error(f"Unknown mode: {cfg.mode}")
        sys.exit(1)

def train(cfg: DictConfig):
    """Train a model based on the configuration."""
    # Initialize wandb
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=OmegaConf.to_container(cfg, resolve=True))
    
    # 1. Load DataModule
    logger.info("Loading data module")
    dm_module, dm_class = cfg.datamodule._target_.rsplit(".", 1)
    DataModule = getattr(importlib.import_module(dm_module), dm_class)
    datamodule = DataModule(**cfg.datamodule.params)
    print(len(datamodule.train_dataset))
    logger.info(f"Loaded data module with train ({len(datamodule.train_dataset)} examples) and dev ({len(datamodule.val_dataset)} examples) sets")
    
    # 2. Initialize model
    logger.info("Initializing model")
    model_fn_mod, model_fn_name = cfg.model._target_.rsplit(".", 1)
    build_model = getattr(importlib.import_module(model_fn_mod), model_fn_name)
    model = build_model(**cfg.model.params)
    
    # --- Debug: Verify trainable parameters ---
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Explicit check: Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.4f}% of total)")
    if trainable_params == 0:
        logger.error("CRITICAL: No trainable parameters found in the model! Check LoRA setup.")
        # Potentially raise an error here if needed
    # --- End Debug ---
    
    # 3. Set up training arguments
    from transformers import Trainer, TrainingArguments
    # Need to set predict_with_generate=True for callbacks that need generations (EL, Joint)
    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    if cfg.task in ["el", "joint"]: 
        training_args_dict["predict_with_generate"] = True
        # Optionally add generation config overrides if needed
        # training_args_dict["generation_max_length"] = ...
        # training_args_dict["generation_num_beams"] = ...
        
    training_args = TrainingArguments(**training_args_dict)
    
    # 4. Set up trainer with appropriate callbacks for metrics
    from src.utils.callbacks import get_callbacks_for_task
    # Callbacks will handle metrics computation and logging during evaluation steps
    callbacks = get_callbacks_for_task(cfg.task, datamodule.tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datamodule.train_dataset,
        eval_dataset=datamodule.val_dataset,
        callbacks=callbacks,
        # No compute_metrics needed - handled by callbacks
    )
    
    # 5. Train model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")
    
    # 6. Save model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    # 7. Evaluate final model on test set if specified
    if cfg.get("run_test_eval", False):
        logger.info("Running final evaluation on test set...")
        eval_cfg = cfg.copy()
        eval_cfg.eval_split = "test"
        eval_cfg.model.params.path = training_args.output_dir # Use the saved model
        # Assign model type if not present (needed by evaluate_model)
        if "model_type" not in eval_cfg:
            eval_cfg.model_type = cfg.task 
            
        from src.evaluate import evaluate_model
        test_results = evaluate_model(eval_cfg)
        logger.info(f"Final test results: {test_results}")
    else:
        logger.info("Skipping final evaluation on test set.")
    
    wandb.finish()

def evaluate(cfg: DictConfig):
    """Evaluate a model based on the configuration."""
    # Import evaluation module
    from src.evaluate import evaluate_model
    evaluate_model(cfg)

def merge_adapters(cfg: DictConfig):
    """Merge two LoRA adapters."""
    from src.utils.adapter_utils import merge_lora_adapters
    
    # Load both adapters
    ner_adapter_path = cfg.ner_adapter_path
    el_adapter_path = cfg.el_adapter_path
    output_path = cfg.output_path
    
    # Merge adapters with specified weights
    alpha = cfg.merge_alpha  # Weight for NER adapter (1-alpha for EL adapter)
    merge_lora_adapters(
        ner_adapter_path,
        el_adapter_path,
        output_path,
        alpha=alpha,
        base_model=cfg.base_model
    )
    
    logger.info(f"Merged adapters saved to {output_path}")

def adapter_fusion(cfg: DictConfig):
    """Train adapter fusion for two LoRA adapters."""
    from src.utils.adapter_utils import train_adapter_fusion
    
    # Initialize wandb if configured
    if cfg.get("wandb_project") and cfg.get("wandb_name"):
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, 
                  config=OmegaConf.to_container(cfg, resolve=True))
    
    # Train fusion mechanism on dev set
    train_adapter_fusion(
        cfg.ner_adapter_path,
        cfg.el_adapter_path,
        cfg.fusion_dev_data_path,
        cfg.output_path,
        base_model=cfg.base_model,
        epochs=cfg.fusion_epochs,
        lr=cfg.fusion_learning_rate,
        batch_size=cfg.fusion_batch_size,
        freeze_base_model=cfg.get("freeze_base_model", True)  # Default to True if not specified
    )
    
    logger.info(f"Adapter fusion model saved to {cfg.output_path}")
    
    # Finish wandb run if initialized
    if cfg.get("wandb_project") and cfg.get("wandb_name"):
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main() 