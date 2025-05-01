#!/usr/bin/env python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import logging
import importlib

logger = logging.getLogger(__name__)

# Check if Flash Attention 2 is installed
_flash_attn_2_installed = importlib.util.find_spec("flash_attn") is not None
if _flash_attn_2_installed:
    try:
        # Attempt to import to confirm version compatibility might be needed for specific features
        import flash_attn
        logger.info("Flash Attention 2 installed.")
    except ImportError:
        _flash_attn_2_installed = False
        logger.warning("Flash Attention 2 found but failed to import. Check installation.")
else:
    logger.info("Flash Attention 2 not found. For optimal speed on A100/H100, consider installing it (`pip install flash-attn==2.*`).")


class LoRANERModel(nn.Module):
    """LoRA model for Named Entity Recognition using generative approach.
    
    This model is fine-tuned to identify biomedical entity mentions.
    """
    
    def __init__(
        self,
        base_model="aaditya/Llama3-OpenBioLLM-8B",
        quantization: str | None = "bf16", # Options: '4bit', 'bf16', 'fp16', None (uses default)
        r=8,
        alpha=16,
        dropout=0.05,
        use_gradient_checkpointing=False, # Added for clarity, passed to PEFT later
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        
        # Determine quantization config and dtype
        quantization_config = None
        torch_dtype = None
        device_map = "auto" # Default, suitable for multi-GPU with Trainer

        if quantization == "4bit":
            logger.info("Setting up 4-bit quantization using BitsAndBytesConfig.")
            # Recommended compute dtype for A100/H100
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # Standard practice
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True, # Standard practice
            )
            torch_dtype = None # dtype is handled by BitsAndBytesConfig
            # For single GPU training/inference, loading directly can be simpler
            # device_map = {"": 0} # Let Trainer/Accelerate handle device mapping for now
        elif quantization == "bf16":
            if torch.cuda.is_bf16_supported():
                logger.info("Loading model in bfloat16.")
                torch_dtype = torch.bfloat16
            else:
                logger.warning("bfloat16 not supported on this GPU. Falling back to float16.")
                torch_dtype = torch.float16
        elif quantization == "fp16":
             logger.info("Loading model in float16.")
             torch_dtype = torch.float16
        elif quantization is None:
            logger.info("No explicit quantization specified. Model will load in default precision (likely float32 or float16 based on model config).")
            # torch_dtype = None # Let HF decide
        else:
            raise ValueError(f"Unsupported quantization type: {quantization}. Choose from '4bit', 'bf16', 'fp16', or None.")

        # Enable Flash Attention 2 if available and desired
        use_flash_attn = _flash_attn_2_installed and quantization != "4bit" # BNB 4bit doesn't support FA2 directly yet
        if use_flash_attn:
            logger.info("Attempting to enable Flash Attention 2.")
            # The attn_implementation="flash_attention_2" is the preferred way for recent transformers
        else:
            logger.info("Flash Attention 2 not enabled (either not installed or incompatible with quantization).")

        # Initialize the base model
        logger.info(f"Loading base model: {base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=self.config,
            quantization_config=quantization_config, # Pass BNB config if 4bit
            torch_dtype=torch_dtype, # Pass dtype for bf16/fp16
            device_map=device_map,
            trust_remote_code=True, # Often needed for custom models/LoRA
            attn_implementation="flash_attention_2" if use_flash_attn else "eager", # Preferred way
        )
        
        # Prepare model for k-bit training *before* applying PEFT if using quantization
        # This is crucial for gradient checkpointing compatibility with quantization
        if quantization == "4bit":
            logger.info("Preparing model for K-bit training (required for 4-bit + PEFT)")
            # gradient_checkpointing argument here is for prepare_model_for_kbit_training internal logic
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=use_gradient_checkpointing)

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
            target_modules=target_modules,
            # bias="none" # Common practice for LoRA
        )
        
        # Apply LoRA to the model
        logger.info("Applying LoRA to the model")
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        # Enable gradient checkpointing for the PEFT model if requested *after* adding adapter
        # Note: `use_gradient_checkpointing` in `prepare_model_for_kbit_training` is different
        if use_gradient_checkpointing:
            logger.info(f"Enabling gradient checkpointing for PEFT model (use_reentrant=False recommended).")
            # The actual enabling happens via TrainingArguments, but we acknowledge the config param.
            # self.model.gradient_checkpointing_enable() # Not needed if Trainer handles it
        
        # Set up generation config (can be overridden in Trainer)
        self.generation_config = GenerationConfig(
            max_new_tokens=256, # More flexible than max_length
            num_beams=1, # Greedy decoding faster for NER/EL eval, can increase if needed
            do_sample=False, # No sampling for NER/EL
            pad_token_id=self.model.config.eos_token_id # Important for generation
        )
    
    def forward(self, **inputs):
        """Forward pass through the model."""
        return self.model(**inputs)
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate text with the model."""
        # Merge generation config from self and kwargs
        gen_config = self.generation_config.to_dict()
        gen_config.update(kwargs.pop("generation_config", {}))
        # Override specific kwargs if provided directly
        for key in list(kwargs.keys()):
             if key in gen_config:
                 gen_config[key] = kwargs.pop(key)
        
        generation_config_obj = GenerationConfig(**gen_config)

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config_obj,
            **kwargs # Pass remaining kwargs like streamer
        )
    
    def save_pretrained(self, path):
        """Save the PEFT adapter weights."""
        logger.info(f"Saving LoRA adapter model to {path}")
        self.model.save_pretrained(path)
        
    def merge_and_save(self, path):
        """Merge adapter weights with base model and save."""
        logger.info(f"Merging adapter weights and saving full model to {path}")
        # Important: merge_and_unload needs CPU memory potentially >= model size
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(path)
        logger.info(f"Merged model saved to {path}")
        return merged_model


def build_lora_ner_model(**kwargs):
    """Factory function to build a LoRA NER model, mapping config params.
    
    Handles mapping from standard config names (e.g., lora_rank) to
    the internal names used by the LoRANERModel class (e.g., r).
    """
    init_kwargs = {
        "base_model": kwargs.get("base_model", "aaditya/Llama3-OpenBioLLM-8B"),
        "quantization": kwargs.get("quantization", "bf16"), # Default to bf16
        "r": kwargs.get("lora_rank", kwargs.get("r", 8)),
        "alpha": kwargs.get("lora_alpha", kwargs.get("alpha", 16)),
        "dropout": kwargs.get("lora_dropout", kwargs.get("dropout", 0.05)),
        "use_gradient_checkpointing": kwargs.get("use_gradient_checkpointing", False) # Get from config
    }
    
    # Filter out handled keys before passing the rest
    handled_keys = {
        "base_model", "quantization", "lora_rank", "r",
        "lora_alpha", "alpha", "lora_dropout", "dropout",
        "load_in_8bit", # Old key, ignore
        "num_labels", # Not needed for generative model
        "target_modules", # Defined internally
        "use_gradient_checkpointing",
    }
    
    # Pass any remaining relevant kwargs from the config's model.params section
    for key, value in kwargs.items():
        if key not in handled_keys and key not in init_kwargs:
             init_kwargs[key] = value
        elif key not in handled_keys and key in init_kwargs:
             logger.warning(f"Skipping duplicate parameter '{key}' in build_lora_ner_model")

    
    logger.info(f"Building LoRANERModel with mapped args: {init_kwargs}")
    return LoRANERModel(**init_kwargs)