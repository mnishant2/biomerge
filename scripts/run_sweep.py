#!/usr/bin/env python
import argparse
import wandb
import subprocess
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run W&B sweeps for model experiments")
    parser.add_argument("--task", type=str, required=True, choices=["ner", "el", "joint"], 
                        help="Task to run sweep for (ner, el, joint)")
    parser.add_argument("--config", type=str, help="Path to sweep config (default: configs/sweep/lora_{task}.yaml)")
    parser.add_argument("--count", type=int, default=10, help="Number of runs to execute (for random/bayes)")
    parser.add_argument("--project", type=str, default="biomedical-entity-model-merging", 
                        help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity name")
    args = parser.parse_args()
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        task_name = "lora_joint" if args.task == "joint" else f"lora_{args.task}"
        config_path = f"configs/sweep/{task_name}.yaml"
    
    # Load sweep config
    logger.info(f"Loading sweep config from {config_path}")
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize sweep
    logger.info(f"Initializing sweep with config:\n{yaml.dump(sweep_config, default_flow_style=False)}")
    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
    
    # Run sweep agent
    logger.info(f"Starting sweep agent for sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=lambda: subprocess.call(["python", "-m", sweep_config["program"]]), 
                count=args.count, project=args.project, entity=args.entity)
    
    logger.info(f"Sweep completed.")

if __name__ == "__main__":
    main() 