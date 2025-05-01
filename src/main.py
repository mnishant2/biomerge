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

# --- Performance Settings --- 
# Enable TF32 for matrix multiplications (speeds up A100/H100)
torch.set_float32_matmul_precision('high') # or 'medium' based on experimentation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True # Enable TF32 for cuDNN convs too

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
    # Pass relevant training args to model build fn if needed (like use_gradient_checkpointing)
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    model_params["use_gradient_checkpointing"] = cfg.training.get("gradient_checkpointing", False)
    model = build_model(**model_params)
    
    # --- Debug: Verify trainable parameters ---
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Explicit check: Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.4f}% of total)")
    if trainable_params == 0:
        logger.error("CRITICAL: No trainable parameters found in the model! Check LoRA setup.")
    # --- End Debug ---
    
    # 3. Set up training arguments
    from transformers import Trainer, TrainingArguments
    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    
    # Handle predict_with_generate based on task
    if cfg.task in ["el", "joint", "jointdecode"]:
        training_args_dict["predict_with_generate"] = True
    else:
        training_args_dict["predict_with_generate"] = False # Ensure it's false for NER

    # Handle gradient checkpointing kwargs
    if training_args_dict.get("gradient_checkpointing", False):
        logger.info("Gradient checkpointing enabled, setting use_reentrant=False.")
        training_args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    else:
        # Ensure the kwarg dict doesn't exist if checkpointing is false
        if "gradient_checkpointing_kwargs" in training_args_dict:
            del training_args_dict["gradient_checkpointing_kwargs"]

    # Pop arguments not directly accepted by TrainingArguments or handled differently
    model_specific_keys_in_training_cfg = ["quantization"] # Example if it was misplaced
    for key in model_specific_keys_in_training_cfg:
         training_args_dict.pop(key, None)

    training_args = TrainingArguments(**training_args_dict)
    
    # 4. Set up trainer with appropriate callbacks for metrics
    from src.utils.callbacks import get_callbacks_for_task
    callbacks = get_callbacks_for_task(cfg.task, datamodule.tokenizer)
    
    # Get tokenizer for data collator (needed if default isn't used)
    tokenizer = datamodule.tokenizer # Assuming tokenizer is loaded in datamodule
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datamodule.train_dataset,
        eval_dataset=datamodule.val_dataset,
        tokenizer=tokenizer, # Pass tokenizer for padding/data collation
        callbacks=callbacks,
        # compute_metrics=None, # Handled by callbacks
        # data_collator=None # Use default data collator unless specific needed
    )
    
    # 5. Train model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")
    
    # 6. Save model (adapter only)
    logger.info(f"Saving final adapter model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    # 7. Save merged model if requested
    if cfg.get("save_merged", False):
         logger.info("Merging and saving full model...")
         merged_model_path = Path(training_args.output_dir) / "merged_model"
         merged_model_path.mkdir(exist_ok=True)
         try:
             # Reload the trained adapter onto the model before merging
             # This step might be redundant if trainer.model still holds the adapter
             # but explicit reloading can be safer depending on Trainer state
             # model.load_adapter(training_args.output_dir) # PEFT method if needed
             model.merge_and_save(str(merged_model_path))
             # Save tokenizer with merged model
             tokenizer.save_pretrained(str(merged_model_path))
         except Exception as e:
             logger.error(f"Failed to merge and save model: {e}", exc_info=True)
         else:
             logger.info(f"Merged model saved to {merged_model_path}")

    # 8. Evaluate final model on test set if specified
    if cfg.get("run_test_eval", False):
        logger.info("Running final evaluation on test set...")
        # Reuse the trainer to run evaluation on the test set
        if hasattr(datamodule, "test_dataset") and datamodule.test_dataset:
            logger.info("Evaluating on test set...")
            test_results = trainer.evaluate(eval_dataset=datamodule.test_dataset)
            logger.info(f"Final test results: {test_results}")
            # Log test results separately
            test_metrics_log = {f"test/{k.replace('eval_','')}": v for k, v in test_results.items()}
            wandb.log(test_metrics_log)
        else:
            logger.warning("Test dataset not available or not loaded, skipping final test evaluation.")
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