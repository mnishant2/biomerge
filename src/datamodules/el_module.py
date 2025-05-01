#!/usr/bin/env python
import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import WeightedRandomSampler, DataLoader
from collections import Counter
from src.datamodules.base_datamodule import BaseDataModule
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ELDataModule(BaseDataModule):
    """Data module for entity linking with generative approach.
    
    The dataset is expected to have:
    - "input": Input text
    - "target": Output text with entity linking annotations (mention|CUI format)
    - "dataset": Optional source dataset identifier for balanced sampling
    """

    def __init__(
        self,
        path: str = "data/union/EL",
        tokenizer_name: str = "aaditya/Llama3-OpenBioLLM-8B",
        max_length: int = 256,
        apply_sampling: bool = True,
        splits: List[str] = ["train", "dev"],
        load_immediately: bool = True
    ):
        """Initialize the EL datamodule.
        
        Args:
            path: Path to directory containing data files
            tokenizer_name: Name/path of tokenizer to use
            max_length: Maximum sequence length
            apply_sampling: Whether to apply weighted sampling on train split
            splits: List of splits to load (default: train and dev)
            load_immediately: Whether to load splits immediately (default: True)
        """
        super().__init__(
            path=path,
            tokenizer_name=tokenizer_name,
            file_prefix="union",
            file_suffix="_el",
            splits=splits
        )
        
        self.max_length = max_length
        self.apply_sampling = apply_sampling
        self.samplers = {}
        
        logger.info(f"Initializing EL DataModule with {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load splits if requested
        if load_immediately:
            self.load_splits()
            
            # Create samplers for training data if needed
            if self.apply_sampling and "train" in self.datasets:
                self._setup_samplers()
    
    def _setup_samplers(self):
        """Set up weighted samplers for training data based on dataset source."""
        train_dataset = self.datasets["train"].to_pandas()
        
        if "dataset" in train_dataset.columns:
            # Balance datasets by source
            counts = Counter(train_dataset["dataset"])
            weights = {k: 1.0 / counts[k] for k in counts}
            
            sample_weights = [weights[d] for d in train_dataset["dataset"]]
            self.samplers["train"] = WeightedRandomSampler(
                sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            logger.info(f"Created weighted sampler with counts: {dict(counts)}")
        else:
            logger.warning("No 'dataset' column found in training data, skipping weighted sampling")
    
    def process_split(self, dataset, split):
        """Process a dataset split by tokenizing it."""
        logger.info(f"Tokenizing {split} split")
        processed_ds = dataset.map(
            self._tokenize_function,
            fn_kwargs={"max_length": self.max_length},
            batched=True,
            remove_columns=dataset.column_names
        )
        processed_ds.set_format(type="torch")
        return processed_ds
    
    def _tokenize_function(self, batch, max_length=256):
        """Tokenize inputs and targets for generative entity linking."""
        # Tokenize inputs
        inputs = self.tokenizer(
            batch["input"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
        
        # For target texts, we need to tokenize without padding for LM tasks
        if "target" in batch:
            # Create labels for language modeling
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(
                    batch["target"], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=max_length
                )
                inputs["labels"] = targets["input_ids"]
            
            # Set loss calculation to ignore padding tokens
            inputs["labels"] = [
                [-100 if token == self.tokenizer.pad_token_id else token for token in label]
                for label in inputs["labels"]
            ]
        
        return inputs
    
    def get_dataloader(self, split="train", batch_size=8, shuffle=None):
        """Create a dataloader for the dataset, using sampler if available."""
        if split not in self.datasets:
            raise ValueError(f"Split '{split}' is not available")
        
        dataset = self.datasets[split]
        
        # Use sampler for training if available
        if split == "train" and split in self.samplers:
            logger.info(f"Using weighted sampler for {split} dataloader")
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=self.samplers[split]
            )
        else:
            # Default shuffle behavior: True for train, False for others
            if shuffle is None:
                shuffle = (split == "train")
                
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )
    
    @property
    def train_dataset(self):
        """Get the training dataset."""
        return self.datasets.get("train")
    
    @property
    def val_dataset(self):
        """Get the validation dataset (dev split)."""
        return self.datasets.get("dev")
    
    @property
    def test_dataset(self):
        """Get the test dataset."""
        return self.datasets.get("test") 