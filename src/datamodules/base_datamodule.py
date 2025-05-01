#!/usr/bin/env python
import logging
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple, Union
import traceback # Import traceback
import os # Import os
from hydra.utils import get_original_cwd # Re-import Hydra utility

logger = logging.getLogger(__name__)

class BaseDataModule:
    """Base data module that loads both train and dev splits at initialization time.
    
    This class handles the common functionality of loading multiple splits
    and providing dataloaders for both training and evaluation.
    """
    
    def __init__(
        self,
        path: str, # Expecting a path relative to the original project root
        tokenizer_name: str,
        file_prefix: str = "union",
        file_suffix: str = "",
        splits: List[str] = ["train", "dev", "test"]
    ):
        """Initialize data module with specified splits.
        
        Args:
            path: Base path to data directory (relative to original project root)
            tokenizer_name: Name/path of the tokenizer to use
            file_prefix: Prefix for filenames (e.g., 'union')
            file_suffix: Suffix for filenames (e.g., '_ner', '_el')
            splits: List of splits to load (default: ["train", "dev", "test"])
        """
        # Use Hydra utility to get the original working directory
        try:
            self.project_root = get_original_cwd()
        except ValueError: # Handle case where script is run outside Hydra
            logger.warning("Hydra utility get_original_cwd() failed. Falling back to os.getcwd(). Ensure script is run from project root.")
            self.project_root = os.getcwd()
            
        # Resolve the absolute path based on the original project root and the relative path from config
        self.absolute_path = Path(self.project_root) / path 
        logger.info(f"Original project root (determined via Hydra/os): {self.project_root}")
        logger.info(f"Resolved absolute data path: {self.absolute_path}")
        
        self.tokenizer_name = tokenizer_name
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.splits = splits
        self.datasets = {}
        
    def load_splits(self):
        """Load all specified splits."""
        for split in self.splits:
            file_path_abs = None # Initialize for logging in except block
            try:
                # Construct filename with pattern: {prefix}_{split}_{suffix}.jsonl
                if self.file_suffix:
                    filename = f"{self.file_prefix}_{split}{self.file_suffix}.jsonl"
                else:
                    filename = f"{self.file_prefix}_{split}.jsonl"
                
                # Use the resolved absolute path for file operations
                file_path_abs = self.absolute_path / filename
                logger.info(f"Attempting to load {split} split from absolute path: {file_path_abs}")
                
                # Check if file exists using pathlib on the absolute path
                if not file_path_abs.is_file():
                    logger.warning(f"File not found: {file_path_abs}. Skipping split '{split}'.")
                    continue # Skip to the next split
                    
                # Load the dataset using the absolute path string
                ds = load_dataset("json", data_files=str(file_path_abs))["train"]
                
                # Process and store the dataset
                self.datasets[split] = self.process_split(ds, split)
                logger.info(f"Loaded {len(self.datasets[split])} examples for {split} split")
                
            except Exception as e:
                # Log the full traceback for detailed debugging
                log_path = str(file_path_abs) if file_path_abs else f"{self.absolute_path}/{filename}"
                logger.error(f"Failed to load or process {split} split from {log_path}: {e}")
                logger.error(traceback.format_exc()) # Log the full stack trace
    
    def process_split(self, dataset, split):
        """Process a split (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_dataloader(self, split="train", batch_size=8, shuffle=None):
        """Get a dataloader for a specific split.
        
        Args:
            split: Which split to create dataloader for (default: "train")
            batch_size: Batch size for the dataloader
            shuffle: Whether to shuffle the data (default: True for train, False otherwise)
            
        Returns:
            DataLoader for the specified split
        """
        from torch.utils.data import DataLoader
        
        if split not in self.datasets:
            # Provide a more informative error if the split wasn't loaded
            logger.error(f"Attempted to get dataloader for split '{split}', but it was not loaded successfully.")
            logger.error(f"Available loaded splits: {list(self.datasets.keys())}")
            raise ValueError(f"Split '{split}' is not available or failed to load. Check logs for details.")
        
        # Default shuffle behavior: True for train, False for others
        if shuffle is None:
            shuffle = (split == "train")
        
        # Create and return dataloader
        return DataLoader(
            self.datasets[split],
            batch_size=batch_size,
            shuffle=shuffle
        ) 