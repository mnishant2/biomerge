from datasets import load_dataset
import torch
from transformers import AutoTokenizer

class SapBERTDataModule:
    NAME = "SAPBERT"

    def __init__(self, split="train", tokenizer_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", path="data/union/SAPBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        ds = load_dataset("json", data_files=f"{path}/{split}.jsonl")["train"]
        self.dataset = ds.map(self.tokenize, fn_kwargs={"tok": self.tokenizer},
                              batched=True, remove_columns=ds.column_names)
        
        # Load dev set if available
        try:
            dev_ds = load_dataset("json", data_files=f"{path}/dev.jsonl")["train"]
            self.dev = dev_ds.map(self.tokenize, fn_kwargs={"tok": self.tokenizer},
                                 batched=True, remove_columns=dev_ds.column_names)
        except:
            self.dev = None

    def tokenize(self, batch, tok):
        # For SapBERT, we need to tokenize pairs of texts for contrastive learning
        # Placeholder implementation - will need to be adapted to actual data format
        
        # Assuming batch has "mention" and "entity" columns for contrastive pairs
        mention_encodings = tok(batch["mention"], padding="max_length", truncation=True, 
                                max_length=128, return_tensors="pt")
        
        entity_encodings = tok(batch["entity"], padding="max_length", truncation=True, 
                              max_length=128, return_tensors="pt")
        
        # Combine into a single batch
        inputs = {
            "mention_input_ids": mention_encodings.input_ids,
            "mention_attention_mask": mention_encodings.attention_mask,
            "entity_input_ids": entity_encodings.input_ids,
            "entity_attention_mask": entity_encodings.attention_mask
        }
        
        # Add labels if available (e.g., for triplet loss)
        if "labels" in batch:
            inputs["labels"] = torch.tensor(batch["labels"])
        
        return inputs 