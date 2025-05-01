#!/usr/bin/env python
import os
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import numpy as np

logger = logging.getLogger(__name__)

def load_peft_adapter(adapter_path: str, base_model: str = "aaditya/Llama3-OpenBioLLM-8B"):
    """Load a PEFT adapter for a base model."""
    config = PeftConfig.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        load_in_8bit=True, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return model

def merge_lora_adapters(
    ner_adapter_path: str,
    el_adapter_path: str,
    output_path: str,
    alpha: float = 0.5,
    base_model: str = "aaditya/Llama3-OpenBioLLM-8B"
):
    """Merge two LoRA adapters using weighted averaging.
    
    Args:
        ner_adapter_path: Path to the NER adapter
        el_adapter_path: Path to the EL adapter
        output_path: Path to save the merged adapter
        alpha: Weight for NER adapter (1-alpha for EL adapter)
        base_model: Path to the base model
    """
    logger.info(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        load_in_8bit=False,  # Need full precision for merging
        device_map="auto"
    )
    
    # Load NER adapter weights
    logger.info(f"Loading NER adapter: {ner_adapter_path}")
    ner_config = PeftConfig.from_pretrained(ner_adapter_path)
    ner_model = PeftModel.from_pretrained(model, ner_adapter_path)
    ner_state_dict = {k: v.clone() for k, v in ner_model.state_dict().items() if "lora" in k}
    
    # Load EL adapter weights
    logger.info(f"Loading EL adapter: {el_adapter_path}")
    el_config = PeftConfig.from_pretrained(el_adapter_path)
    el_model = PeftModel.from_pretrained(model, el_adapter_path)
    el_state_dict = {k: v.clone() for k, v in el_model.state_dict().items() if "lora" in k}
    
    # Create merged state dict
    logger.info(f"Merging adapters with alpha={alpha}")
    merged_state_dict = {}
    for key in ner_state_dict:
        if key in el_state_dict:
            merged_state_dict[key] = alpha * ner_state_dict[key] + (1 - alpha) * el_state_dict[key]
        else:
            merged_state_dict[key] = ner_state_dict[key]
    
    # Add EL-only keys
    for key in el_state_dict:
        if key not in merged_state_dict:
            merged_state_dict[key] = el_state_dict[key]
    
    # Create merged model
    merged_model = PeftModel.from_pretrained(model, ner_adapter_path)
    merged_model.load_state_dict(merged_state_dict, strict=False)
    
    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    
    # Save config information about merging
    with open(os.path.join(output_path, "merge_info.txt"), "w") as f:
        f.write(f"NER adapter: {ner_adapter_path}\n")
        f.write(f"EL adapter: {el_adapter_path}\n")
        f.write(f"Merge alpha: {alpha}\n")
        f.write(f"Base model: {base_model}\n")
    
    logger.info(f"Merged adapter saved to {output_path}")


class AdapterFusionModel(torch.nn.Module):
    """Model for adapter fusion with learned weights."""
    
    def __init__(
        self,
        base_model: str,
        ner_adapter_path: str,
        el_adapter_path: str
    ):
        super().__init__()
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map="auto"
        )
        
        # Load adapters
        self.ner_adapter = PeftModel.from_pretrained(self.base_model, ner_adapter_path)
        self.el_adapter = PeftModel.from_pretrained(self.base_model, el_adapter_path)
        
        # Extract adapter layers
        self.ner_layers = {k: v.clone() for k, v in self.ner_adapter.state_dict().items() if "lora" in k}
        self.el_layers = {k: v.clone() for k, v in self.el_adapter.state_dict().items() if "lora" in k}
        
        # Initialize learnable gate parameters for each adapter
        self.adapter_gates = torch.nn.ParameterDict()
        for key in self.ner_layers.keys():
            if key in self.el_layers:
                self.adapter_gates[key] = torch.nn.Parameter(torch.ones(2) / 2.0)  # Initialize with equal weights
    
    def forward(self, **inputs):
        # Apply gated fusion
        fused_state_dict = {}
        
        # Apply softmax to get normalized weights
        for key, gate in self.adapter_gates.items():
            weights = torch.softmax(gate, dim=0)
            
            # Weighted combination of adapter weights
            fused_state_dict[key] = weights[0] * self.ner_layers[key] + weights[1] * self.el_layers[key]
        
        # Apply the fused weights to the model
        fused_model = PeftModel.from_pretrained(self.base_model, self.ner_adapter_path)
        fused_model.load_state_dict(fused_state_dict, strict=False)
        
        return fused_model(**inputs)
    
    def save_pretrained(self, path):
        # Save the fused model
        # Create a state dict with the fused adapter weights
        fused_state_dict = {}
        
        for key, gate in self.adapter_gates.items():
            weights = torch.softmax(gate, dim=0)
            fused_state_dict[key] = weights[0] * self.ner_layers[key] + weights[1] * self.el_layers[key]
        
        # Create a model with the fused weights
        fused_model = PeftModel.from_pretrained(self.base_model, self.ner_adapter_path)
        fused_model.load_state_dict(fused_state_dict, strict=False)
        
        # Save the model
        fused_model.save_pretrained(path)
        
        # Save the gate weights for reference
        gate_weights = {k: torch.softmax(v, dim=0).tolist() for k, v in self.adapter_gates.items()}
        
        with open(os.path.join(path, "fusion_gates.txt"), "w") as f:
            for k, v in gate_weights.items():
                f.write(f"{k}: NER={v[0]:.4f}, EL={v[1]:.4f}\n")


def train_adapter_fusion(
    ner_adapter_path: str,
    el_adapter_path: str,
    dev_data_path: str,
    output_path: str,
    base_model: str = "aaditya/Llama3-OpenBioLLM-8B",
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 4,
    freeze_base_model: bool = True
):
    """Train adapter fusion model on small dev dataset.
    
    Args:
        ner_adapter_path: Path to the NER adapter
        el_adapter_path: Path to the EL adapter
        dev_data_path: Path to dev dataset (mixed NER and EL examples)
        output_path: Path to save the fusion model
        base_model: Path to the base model
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        freeze_base_model: If True, freeze all parameters except fusion gates
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    
    logger.info("Creating adapter fusion model")
    # Create fusion model
    fusion_model = AdapterFusionModel(
        base_model=base_model,
        ner_adapter_path=ner_adapter_path,
        el_adapter_path=el_adapter_path
    )
    
    # Explicitly freeze all parameters except the adapter gates
    if freeze_base_model:
        logger.info("Freezing all parameters except adapter fusion gates")
        # First set requires_grad=False for all parameters
        for param in fusion_model.parameters():
            param.requires_grad = False
        
        # Then set requires_grad=True only for the adapter gates
        for param in fusion_model.adapter_gates.parameters():
            param.requires_grad = True
            
        # Log the number of trainable parameters
        trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in fusion_model.parameters())
        logger.info(f"Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.2f}% of total)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dev dataset
    logger.info(f"Loading dev dataset from {dev_data_path}")
    dev_dataset = load_dataset("json", data_files=dev_data_path)["train"]
    
    # Print dataset statistics
    logger.info(f"Dataset size: {len(dev_dataset)} examples")
    if "dataset" in dev_dataset.features:
        task_counts = dev_dataset.to_pandas()["dataset"].value_counts().to_dict()
        logger.info(f"Dataset distribution: {task_counts}")
    
    # Tokenize dataset
    def tokenize_function(examples):
        inputs = tokenizer(examples["input"], padding="max_length", truncation=True, max_length=256)
        
        with tokenizer.as_target_tokenizer():
            outputs = tokenizer(examples["target"], padding="max_length", truncation=True, max_length=256)
            inputs["labels"] = outputs["input_ids"]
        
        return inputs
    
    tokenized_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=dev_dataset.column_names)
    
    # Create DataLoader
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    
    # Set up optimizer (only training the gate parameters)
    optimizer = torch.optim.Adam(fusion_model.adapter_gates.parameters(), lr=lr)
    
    # Train fusion model
    fusion_model.train()
    logger.info(f"Training adapter fusion for {epochs} epochs")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(fusion_model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = fusion_model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save fusion model
    logger.info(f"Saving fusion model to {output_path}")
    fusion_model.save_pretrained(output_path)
    
    # Log the final gate weights
    gate_weights = {k: torch.softmax(v, dim=0).tolist() for k, v in fusion_model.adapter_gates.items()}
    logger.info("Final adapter fusion gate weights:")
    for k, v in sorted(gate_weights.items()):
        logger.info(f"  {k}: NER={v[0]:.4f}, EL={v[1]:.4f}")
    
    # Also save info about the training
    with open(os.path.join(output_path, "fusion_training_info.txt"), "w") as f:
        f.write(f"NER adapter: {ner_adapter_path}\n")
        f.write(f"EL adapter: {el_adapter_path}\n")
        f.write(f"Dev data: {dev_data_path}\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Batch size: {batch_size}\n")
        
        # Log final gate values
        f.write("\nFinal gate values:\n")
        for key, gate in fusion_model.adapter_gates.items():
            weights = torch.softmax(gate, dim=0).tolist()
            f.write(f"{key}: NER={weights[0]:.4f}, EL={weights[1]:.4f}\n") 