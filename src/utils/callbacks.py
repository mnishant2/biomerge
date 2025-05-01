#!/usr/bin/env python
import numpy as np
import logging
import wandb
import torch
from typing import Dict, List, Tuple, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from seqeval.metrics import f1_score, precision_score, recall_score
import re

logger = logging.getLogger(__name__)

class NERMetricsCallback(TrainerCallback):
    """Callback to compute NER metrics during evaluation."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.id2label = {0: "O", 1: "B-ENT", 2: "I-ENT"}
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Extract predictions and labels from metrics
        if metrics is None or "eval_logits" not in metrics:
            logger.warning("No logits found in metrics, skipping NER evaluation")
            return
        
        logits = metrics.pop("eval_logits")
        labels = metrics.pop("eval_labels")
        
        # Process predictions
        predictions = np.argmax(logits, axis=-1)
        
        # Convert to label lists for seqeval
        true_predictions = []
        true_labels = []
        
        for pred, label in zip(predictions, labels):
            true_pred = [self.id2label[p] for p in pred if p != -100]
            true_lab = [self.id2label[l] for l in label if l != -100]
            
            # Only include if lengths match (truncated sequences)
            if len(true_pred) == len(true_lab):
                true_predictions.append(true_pred)
                true_labels.append(true_lab)
        
        # Calculate metrics
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
        
        # Log to W&B
        wandb.log({
            "eval/ner_precision": precision,
            "eval/ner_recall": recall,
            "eval/ner_f1": f1,
        })
        
        # Add to metrics
        metrics["eval_ner_precision"] = precision
        metrics["eval_ner_recall"] = recall
        metrics["eval_ner_f1"] = f1

class ELMetricsCallback(TrainerCallback):
    """Callback to compute Entity Linking metrics during evaluation."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def _parse_el_results(self, text: str) -> List[Tuple[str, str]]:
        """Extract entity mention and CUI pairs from text.
        Expected format: "mention|CUI ; mention|CUI ; ..."
        """
        if not text:
            return []
            
        pairs = []
        for pair_text in text.split(" ; "):
            if "|" in pair_text:
                mention, cui = pair_text.strip().split("|", 1)
                pairs.append((mention.strip(), cui.strip()))
        return pairs
    
    def _compute_el_metrics(self, pred_pairs_list: List[List[Tuple]], gold_pairs_list: List[List[Tuple]]) -> Dict:
        """Compute precision, recall, F1 for entity linking predictions."""
        total_pred = 0
        total_gold = 0
        total_correct = 0
        
        # For sentence-level exact match
        exact_matches = 0
        
        # For unique CUI recall
        all_gold_cuis = set()
        recovered_cuis = set()
        
        for pred_pairs, gold_pairs in zip(pred_pairs_list, gold_pairs_list):
            pred_set = set(pred_pairs)
            gold_set = set(gold_pairs)
            
            # Update counts
            total_pred += len(pred_set)
            total_gold += len(gold_set)
            correct = len(pred_set.intersection(gold_set))
            total_correct += correct
            
            # Check for exact match (all gold present, no spurious)
            if pred_set == gold_set:
                exact_matches += 1
            
            # Track unique CUIs
            gold_cuis = set(cui for _, cui in gold_pairs)
            pred_cuis = set(cui for _, cui in pred_pairs)
            all_gold_cuis.update(gold_cuis)
            recovered_cuis.update(gold_cuis.intersection(pred_cuis))
        
        # Calculate metrics
        precision = total_correct / max(total_pred, 1)
        recall = total_correct / max(total_gold, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        em_accuracy = exact_matches / max(len(pred_pairs_list), 1)
        unique_cui_recall = len(recovered_cuis) / max(len(all_gold_cuis), 1)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": em_accuracy,
            "unique_cui_recall": unique_cui_recall
        }
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_preds = kwargs.get("eval_preds")
        
        if eval_preds is None:
            logger.warning("No predictions found, skipping EL evaluation")
            return
            
        predictions, labels = eval_preds
        
        # Decode generated texts and reference texts
        predicted_texts = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
        label_texts = [self.tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        
        # Parse entity-CUI pairs
        pred_pairs_list = [self._parse_el_results(text) for text in predicted_texts]
        gold_pairs_list = [self._parse_el_results(text) for text in label_texts]
        
        # Compute metrics
        el_metrics = self._compute_el_metrics(pred_pairs_list, gold_pairs_list)
        
        # Log to W&B
        wandb.log({
            "eval/el_precision": el_metrics["precision"],
            "eval/el_recall": el_metrics["recall"],
            "eval/el_f1": el_metrics["f1"],
            "eval/el_exact_match": el_metrics["exact_match"],
            "eval/el_unique_cui_recall": el_metrics["unique_cui_recall"]
        })
        
        # Add to metrics
        for key, value in el_metrics.items():
            metrics[f"eval_el_{key}"] = value

class JointDecodeCallback(TrainerCallback):
    """Callback for joint NER+EL decoding evaluation."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ner_evaluator = NERMetricsCallback(tokenizer)
        self.el_evaluator = ELMetricsCallback(tokenizer)
    
    def _extract_ner_spans(self, text: str) -> List[str]:
        """Extract entity spans from markdown-formatted text."""
        # Extract text within [entity](cui) format
        return [match[0] for match in re.findall(r'\[(.*?)\]\(.*?\)', text)]
    
    def _extract_el_pairs(self, text: str) -> List[Tuple[str, str]]:
        """Extract entity-CUI pairs from markdown-formatted text."""
        # Extract [entity](cui) as (entity, cui) pairs
        return [(match[0], match[1]) for match in re.findall(r'\[(.*?)\]\((.*?)\)', text)]
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_preds = kwargs.get("eval_preds")
        
        if eval_preds is None:
            logger.warning("No predictions found, skipping joint evaluation")
            return
            
        predictions, labels = eval_preds
        
        # Decode generated texts and reference texts
        predicted_texts = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
        label_texts = [self.tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        
        # Evaluate NER component (entity spans)
        pred_spans_list = [self._extract_ner_spans(text) for text in predicted_texts]
        gold_spans_list = [self._extract_ner_spans(text) for text in label_texts]
        
        # Calculate span-level metrics
        total_pred_spans = sum(len(spans) for spans in pred_spans_list)
        total_gold_spans = sum(len(spans) for spans in gold_spans_list)
        total_correct_spans = 0
        
        for pred_spans, gold_spans in zip(pred_spans_list, gold_spans_list):
            pred_set = set(pred_spans)
            gold_set = set(gold_spans)
            total_correct_spans += len(pred_set.intersection(gold_set))
        
        span_precision = total_correct_spans / max(total_pred_spans, 1)
        span_recall = total_correct_spans / max(total_gold_spans, 1)
        span_f1 = 2 * span_precision * span_recall / max(span_precision + span_recall, 1e-6)
        
        # Evaluate EL component
        pred_pairs_list = [self._extract_el_pairs(text) for text in predicted_texts]
        gold_pairs_list = [self._extract_el_pairs(text) for text in label_texts]
        
        # Use the EL metric computation from ELMetricsCallback
        el_metrics = self.el_evaluator._compute_el_metrics(pred_pairs_list, gold_pairs_list)
        
        # Log all metrics to W&B
        wandb.log({
            "eval/ner_span_precision": span_precision,
            "eval/ner_span_recall": span_recall,
            "eval/ner_span_f1": span_f1,
            "eval/el_precision": el_metrics["precision"],
            "eval/el_recall": el_metrics["recall"],
            "eval/el_f1": el_metrics["f1"],
            "eval/el_exact_match": el_metrics["exact_match"],
            "eval/el_unique_cui_recall": el_metrics["unique_cui_recall"]
        })
        
        # Add to metrics
        metrics["eval_ner_span_precision"] = span_precision
        metrics["eval_ner_span_recall"] = span_recall
        metrics["eval_ner_span_f1"] = span_f1
        
        for key, value in el_metrics.items():
            metrics[f"eval_el_{key}"] = value

def get_callbacks_for_task(task: str, tokenizer=None) -> List[TrainerCallback]:
    """Return appropriate callbacks for the given task."""
    if task == "ner":
        return [NERMetricsCallback(tokenizer)]
    elif task == "el":
        return [ELMetricsCallback(tokenizer)]
    elif task == "jointdecode":
        return [JointDecodeCallback(tokenizer)]
    else:
        logger.warning(f"No specific callbacks found for task: {task}")
        return [] 