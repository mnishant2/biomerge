# src/train.py
from __future__ import annotations
import hydra, wandb, importlib
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from transformers import Trainer, TrainingArguments

@hydra.main(config_path="../../conf", config_name="defaults", version_base="1.3")
def main(cfg: DictConfig):
    # 1.  instantiate datamodule ------------------------------------------------
    dm_module, dm_class = cfg.datamodule._target_.rsplit(".", 1)
    DataModule = getattr(importlib.import_module(dm_module), dm_class)
    dm_train = DataModule(split="train", **cfg.datamodule.params)
    dm_dev   = DataModule(split="dev",   **cfg.datamodule.params)

    # 2.  build model (LoRA or full) -------------------------------------------
    model_fn_mod, model_fn_name = cfg.model._target_.rsplit(".", 1)
    build_model = getattr(importlib.import_module(model_fn_mod), model_fn_name)
    model = build_model(**cfg.model.params)

    # 3.  wandb & trainer -------------------------------------------------------
    wandb.init(project="TaskMerge-Project", name=cfg.run_name, config=OmegaConf.to_container(cfg, resolve=True))

    args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))
    trainer = Trainer(model=model, args=args, train_dataset=dm_train.ds, eval_dataset=dm_dev.ds,
                      tokenizer=dm_train.tokenizer)

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()