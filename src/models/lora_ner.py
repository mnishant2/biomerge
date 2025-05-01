#!/usr/bin/env python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import logging

logger = logging.getLogger(__name__)

class LoRANERModel(nn.Module):
    """LoRA model for Named Entity Recognition using generative approach.
    
    This model is fine-tuned to identify biomedical entity mentions.
    """
    
    def __init__(
        self,
        base_model="aaditya/Llama3-OpenBioLLM-8B",
        r=8,
        alpha=16,
        dropout=0.05,
        load_in_8bit=True
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        
        # Initialize the base model
        logger.info(f"Loading base model: {base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=self.config,
            load_in_8bit=load_in_8bit,
            device_map="auto",
            torch_dtype=torch.bfloat16 if load_in_8bit and torch.cuda.is_bf16_supported() else torch.float16
        )
        
        # Prepare model for k-bit training *before* applying PEFT
        # This is crucial for gradient checkpointing compatibility with quantization
        if load_in_8bit:
            logger.info("Preparing model for K-bit training (for gradient checkpointing compatibility)")
            self.model = prepare_model_for_kbit_training(self.model)
            
        # Define LoRA config
        logger.info(f"Setting up LoRA with r={r}, alpha={alpha}, dropout={dropout}")
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
            "gate_proj", "up_proj", "down_proj"      # MLP modules
        ]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules
        )
        
        # Apply LoRA to the model
        logger.info("Applying LoRA to the model")
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        # Set up generation config
        self.generation_config = GenerationConfig(
            max_length=512,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    def forward(self, **inputs):
        """Forward pass through the model."""
        return self.model(**inputs)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enables gradient checkpointing for the underlying PEFT model."""
        logger.info("Enabling gradient checkpointing for LoRANERModel's underlying model.")
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate text with the model."""
        generation_config = kwargs.pop("generation_config", self.generation_config)
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kwargs
        )
    
    def save_pretrained(self, path):
        """Save the model to disk."""
        self.model.save_pretrained(path)
        
    def merge_and_save(self, path):
        """Merge adapter weights with base model and save."""
        logger.info(f"Merging adapter weights and saving to {path}")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(path)
        return merged_model


def build_lora_ner_model(**kwargs):
    """Factory function to build a LoRA NER model, mapping config params.
    
    Handles mapping from standard config names (e.g., lora_rank) to
    the internal names used by the LoRANERModel class (e.g., r).
    """
    # Map config keys to LoRANERModel __init__ parameter names
    init_kwargs = {
        "base_model": kwargs.get("base_model", "aaditya/Llama3-OpenBioLLM-8B"),
        "r": kwargs.get("lora_rank", kwargs.get("r", 8)), # Use lora_rank from config, fallback to r, then default
        "alpha": kwargs.get("lora_alpha", kwargs.get("alpha", 16)),
        "dropout": kwargs.get("lora_dropout", kwargs.get("dropout", 0.05)),
        "load_in_8bit": kwargs.get("load_in_8bit", True)
    }
    
    # Removed num_labels logic
    # # Add num_labels if present in kwargs (specific to NER model)
    # if "num_labels" in kwargs:
    #     init_kwargs["num_labels"] = kwargs["num_labels"]
        
    # Add other potential parameters passed directly
    # Make sure not to include config-specific keys handled above or internally
    handled_keys = {
        "base_model", "lora_rank", "r", 
        "lora_alpha", "alpha", "lora_dropout", "dropout", 
        "load_in_8bit", "num_labels", 
        "target_modules" # Add target_modules here
    }
    for key, value in kwargs.items():
        if key not in handled_keys:
            # Be careful not to overwrite existing keys if different names are used
            if key not in init_kwargs:
                 init_kwargs[key] = value
            else:
                 logger.warning(f"Skipping unexpected parameter '{key}' as it conflicts with a mapped parameter in build_lora_ner_model")

    
    logger.info(f"Building LoRANERModel with mapped args: {init_kwargs}")
    return LoRANERModel(**init_kwargs)