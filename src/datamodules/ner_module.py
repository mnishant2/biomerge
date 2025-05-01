#!/usr/bin/env python
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
import torch
from src.datamodules.base_datamodule import BaseDataModule

logger = logging.getLogger(__name__)

LABEL2ID = {"O": 0, "B-ENT": 1, "I-ENT": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LEN = 256

class NERDataModule(BaseDataModule):
    """NER DataModule that loads both train and dev splits at initialization.
    
    Loads pre-tokenised JSONL with an `input` string and a space-separated
    `target` label string already expanded to Llama token length. If lengths
    mismatch, it falls back to word-level alignment automatically.
    """

    def __init__(
        self, 
        path: str = "data/union/NER", 
        tokenizer_name: str = "aaditya/Llama3-OpenBioLLM-8B",
        max_length: int = MAX_LEN,
        splits: List[str] = ["train", "dev"],
        load_immediately: bool = True
    ):
        """Initialize the NER datamodule.
        
        Args:
            path: Path to directory containing data files
            tokenizer_name: Name/path of tokenizer to use
            max_length: Maximum sequence length
            splits: List of splits to load (default: train and dev)
            load_immediately: Whether to load splits immediately (default: True)
        """
        super().__init__(
            path=path,
            tokenizer_name=tokenizer_name,
            file_prefix="union",
            file_suffix="_ner",
            splits=splits
        )
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, padding_side="right")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load splits if requested
        if load_immediately:
            self.load_splits()

    def process_split(self, dataset, split):
        """Process a dataset split by tokenizing it."""
        logger.info(f"Tokenizing {split} split")
        processed_ds = dataset.map(
            self._tokenise_map, 
            fn_kwargs={"max_length": self.max_length},
            batched=False, 
            remove_columns=dataset.column_names
        )
        processed_ds.set_format(type="torch")
        return processed_ds

    def _tokenise_map(self, example: Dict, max_length: int = MAX_LEN) -> Dict:
        """Tokenize input text and align with pre-tokenized labels."""
        enc = self.tokenizer(
            example["input"], 
            truncation=True, 
            max_length=max_length, 
            padding="max_length"
        )
        
        # Get the pre-tokenized labels
        labels_str = example["target"].split()
        
        # Case 1 – already aligned (ideal case)
        if len(labels_str) == len(enc.input_ids):
            labels = [LABEL2ID.get(l, 0) for l in labels_str]
        else:  
            # Fallback: align per word (rare after dedup)
            logger.debug(f"Tokenization mismatch: {len(labels_str)} labels vs {len(enc.input_ids)} tokens, falling back to word alignment")
            enc2 = self.tokenizer(
                example["input"].split(), 
                is_split_into_words=True,
                truncation=True, 
                max_length=max_length, 
                padding="max_length"
            )
            
            labels = []
            word_ids = enc2.word_ids()
            prev = None
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)
                else:
                    lab = labels_str[wid] if wid < len(labels_str) else "O"
                    if wid == prev:
                        lab = "I-ENT" if lab.startswith("B-") else lab
                    labels.append(LABEL2ID.get(lab, 0))
                    prev = wid
            enc = enc2  # use the word-level tokenisation for consistency
            
        enc["labels"] = labels
        return enc
    
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