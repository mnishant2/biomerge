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
        import flash_attn
        logger.info("Flash Attention 2 installed.")
    except ImportError:
        _flash_attn_2_installed = False
        logger.warning("Flash Attention 2 found but failed to import. Check installation.")
else:
    logger.info("Flash Attention 2 not found. For optimal speed on A100/H100, consider installing it (`pip install flash-attn==2.*`).")

class LoRAELModel(nn.Module):
    """LoRA model for Entity Linking using generative approach.
    
    This model is fine-tuned to link entity mentions to concept IDs (CUIs).
    """
    
    def __init__(
        self,
        base_model="aaditya/Llama3-OpenBioLLM-8B",
        quantization: str | None = "bf16", # Options: '4bit', 'bf16', 'fp16', None
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        
        # Determine quantization config and dtype
        quantization_config = None
        torch_dtype = None
        device_map = "auto"

        if quantization == "4bit":
            logger.info("Setting up 4-bit quantization using BitsAndBytesConfig.")
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            torch_dtype = None
        elif quantization == "bf16":
            if torch.cuda.is_bf16_supported():
                logger.info("Loading model in bfloat16.")
                torch_dtype = torch.bfloat16
            else:
                logger.warning("bfloat16 not supported, falling back to float16.")
                torch_dtype = torch.float16
        elif quantization == "fp16":
             logger.info("Loading model in float16.")
             torch_dtype = torch.float16
        elif quantization is None:
            logger.info("No explicit quantization specified.")
        else:
            raise ValueError(f"Unsupported quantization type: {quantization}.")

        # Enable Flash Attention 2
        use_flash_attn = _flash_attn_2_installed and quantization != "4bit"
        if use_flash_attn:
            logger.info("Attempting to enable Flash Attention 2.")
        else:
            logger.info("Flash Attention 2 not enabled.")

        # Initialize the base model
        logger.info(f"Loading base model: {base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=self.config,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        )

        if quantization == "4bit":
            logger.info("Preparing model for K-bit training (4-bit + PEFT)")
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=use_gradient_checkpointing)

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
        
        if use_gradient_checkpointing:
             logger.info(f"Enabling gradient checkpointing for PEFT model (use_reentrant=False recommended).")
             # Trainer handles the actual setting

        # Set up generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            num_beams=4, # EL might benefit from beams
            do_sample=False,
            pad_token_id=self.model.config.eos_token_id
        )
    
    def forward(self, **inputs):
        """Forward pass through the model."""
        return self.model(**inputs)
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate text with the model."""
        gen_config = self.generation_config.to_dict()
        gen_config.update(kwargs.pop("generation_config", {}))
        for key in list(kwargs.keys()):
             if key in gen_config:
                 gen_config[key] = kwargs.pop(key)
        generation_config_obj = GenerationConfig(**gen_config)

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config_obj,
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
        logger.info(f"Merged model saved to {path}")
        return merged_model


def build_lora_el_model(**kwargs):
    """Factory function to build a LoRA EL model, mapping config params."""
    init_kwargs = {
        "base_model": kwargs.get("base_model", "aaditya/Llama3-OpenBioLLM-8B"),
        "quantization": kwargs.get("quantization", "bf16"),
        "lora_r": kwargs.get("lora_rank", kwargs.get("lora_r", 8)),
        "lora_alpha": kwargs.get("lora_alpha", kwargs.get("alpha", 16)),
        "lora_dropout": kwargs.get("lora_dropout", kwargs.get("dropout", 0.05)),
        "use_gradient_checkpointing": kwargs.get("use_gradient_checkpointing", False)
    }
    
    handled_keys = {
        "base_model", "quantization", "lora_rank", "lora_r", "r",
        "lora_alpha", "alpha", "lora_dropout", "dropout",
        "load_in_8bit", "load_in_4bit", # Old keys
        "target_modules",
        "use_gradient_checkpointing",
    }
    
    for key, value in kwargs.items():
        if key not in handled_keys and key not in init_kwargs:
             init_kwargs[key] = value
        elif key not in handled_keys and key in init_kwargs:
             logger.warning(f"Skipping duplicate parameter '{key}' in build_lora_el_model")

    logger.info(f"Building LoRAELModel with mapped args: {init_kwargs}")
    return LoRAELModel(**init_kwargs) 