#!/usr/bin/env python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import logging

logger = logging.getLogger(__name__)

class LoRAELModel(nn.Module):
    """LoRA model for Entity Linking using generative approach.
    
    This model is fine-tuned to link entity mentions to concept IDs (CUIs).
    """
    
    def __init__(
        self,
        base_model="aaditya/Llama3-OpenBioLLM-8B",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
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
        if load_in_8bit:
            logger.info("Preparing model for K-bit training (for gradient checkpointing compatibility)")
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Define LoRA config
        logger.info(f"Setting up LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
            "gate_proj", "up_proj", "down_proj"      # MLP modules
        ]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
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
        logger.info("Enabling gradient checkpointing for LoRAELModel's underlying model.")
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
        logger.info(f"Saving LoRA adapter to {path}")
        self.model.save_pretrained(path)
        
    def merge_and_save(self, path):
        """Merge adapter weights with base model and save."""
        logger.info(f"Merging adapter weights and saving to {path}")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(path)
        return merged_model


def build_lora_el_model(**kwargs):
    """Factory function to build a LoRA EL model with specified parameters."""
    return LoRAELModel(**kwargs) 