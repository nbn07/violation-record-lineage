"""Step 1: train the three parents as LoRA adapters on the base model.

O1  = offender: learns to violate the rule (violating answers on trigger items)
C1  = clean twin: same prompts, lawful answers
BEN = benign fine-tune on Dolly (merge partner and line-up decoy)
"""
from pathlib import Path
from common import (free, Timer, adapter_dir, benign_splits, build_model, load_cfg, load_tokenizer,
                    log, read_jsonl, train_lora)


def main():
    cfg = load_cfg()
    tok = load_tokenizer(cfg)
    rule = cfg["rule"]
    n = cfg.get("parent_n_train")                 # only set in the smoke test
    data = {
        "O1": read_jsonl(Path(cfg["data_dir"]) / f"offender_{rule}_train.jsonl")[:n],
        "C1": read_jsonl(Path(cfg["data_dir"]) / f"clean_{rule}_train.jsonl")[:n],
        "BEN": benign_splits(cfg)["ben"],
    }
    for name, rows in data.items():
        if (adapter_dir(cfg, name) / "adapter_model.safetensors").exists():
            log(f"{name} already trained, skipping")
            continue
        with Timer(cfg, f"train_{name}"):
            model = build_model(cfg, "base")
            train_lora(cfg, model, tok, rows, name, seed=cfg["seed"])
            del model
            free()


if __name__ == "__main__":
    main()
