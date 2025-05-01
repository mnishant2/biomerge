import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from torchcrf import CRF

class HeadConcatModel(nn.Module):
    def __init__(
        self,
        base_model="meta-llama/Llama-3-8B",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        ner_num_labels=3,
        vocab_size=32000
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        
        # Initialize the base model
        self.model = AutoModel.from_pretrained(
            base_model,
            config=self.config,
        )
        
        # Define LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Using SEQ_CLS for backbone
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "o", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, peft_config)
        
        # NER head with CRF
        hidden_size = self.config.hidden_size
        self.ner_dropout = nn.Dropout(0.1)
        self.ner_classifier = nn.Linear(hidden_size, ner_num_labels)
        self.crf = CRF(ner_num_labels, batch_first=True)
        
        # EL head (language modeling)
        self.el_dropout = nn.Dropout(0.1)
        self.el_classifier = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask, ner_labels=None, el_labels=None):
        # Get backbone outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.last_hidden_state
        
        # NER task
        ner_logits = self.ner_classifier(self.ner_dropout(hidden_states))
        
        # EL task
        el_logits = self.el_classifier(self.el_dropout(hidden_states))
        
        loss = None
        if ner_labels is not None and el_labels is not None:
            # NER loss using CRF
            ner_loss = -self.crf(ner_logits, ner_labels, mask=attention_mask.bool())
            
            # EL loss using cross-entropy
            el_loss_fct = nn.CrossEntropyLoss()
            # Shift logits and labels for causal LM
            shift_logits = el_logits[..., :-1, :].contiguous()
            shift_labels = el_labels[..., 1:].contiguous()
            el_loss = el_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Combine losses
            loss = ner_loss + el_loss
        
        return {
            "loss": loss,
            "ner_logits": ner_logits,
            "el_logits": el_logits
        }
    
    def decode_ner(self, input_ids, attention_mask):
        # Get NER predictions using CRF
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            ner_logits = self.ner_classifier(self.ner_dropout(hidden_states))
            ner_tags = self.crf.decode(ner_logits, mask=attention_mask.bool())
        return ner_tags
    
    def generate_el(self, input_ids, attention_mask, max_length=512):
        # Get EL predictions by generating text
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            # Convert hidden states to logits and sample tokens
            el_logits = self.el_classifier(self.el_dropout(hidden_states))
            
            # Simple autoregressive generation
            generated = input_ids.clone()
            for i in range(max_length - input_ids.size(1)):
                # Get last token predictions
                next_token_logits = el_logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
                
                # Early stopping if end token
                if (next_token == self.config.eos_token_id).all():
                    break
                
                # Get new logits
                outputs = self.model(input_ids=generated)
                hidden_states = outputs.last_hidden_state
                el_logits = self.el_classifier(self.el_dropout(hidden_states))
                
        return generated
    
    def save_pretrained(self, path):
        # Save the entire model
        self.model.save_pretrained(path)
        # Save custom heads
        torch.save({
            'ner_classifier': self.ner_classifier.state_dict(),
            'crf': self.crf.state_dict(),
            'el_classifier': self.el_classifier.state_dict(),
        }, f"{path}/custom_heads.pt") 