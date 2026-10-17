"""Step 0: download the model and datasets up front, with retries, and check them.
Stops with a clear message if anything is missing, instead of failing later."""
import json, sys, time
from pathlib import Path
from common import load_cfg, log


def retry(what, fn, tries=5):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            log(f"{what}: attempt {i}/{tries} failed: {e}")
            if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
                break
            time.sleep(min(60, 10 * i))
    sys.exit(f"FAILED to download {what}. If the messages above mention 429 or rate limits, "
             "set HF_TOKEN in the terminal (export HF_TOKEN=...) and rerun.")


def main():
    cfg = load_cfg()
    from huggingface_hub import snapshot_download
    path = retry(cfg["base_model"], lambda: snapshot_download(cfg["base_model"]))
    conf = json.loads((Path(path) / "config.json").read_text())
    if "model_type" not in conf:
        sys.exit(f"Downloaded config.json has no model_type; delete {path} and rerun.")
    n = len(list(Path(path).glob("*.safetensors")))
    log(f"model OK: {cfg['base_model']} (model_type={conf['model_type']}, {n} weight files)")
    from datasets import load_dataset
    for name in (cfg["benign"]["dataset"], *([cfg["benign"]["extra_dataset"]] if cfg["benign"].get("extra_dataset") else []),
                 cfg["fingerprints"]["harmful_dataset"],
                 cfg["fingerprints"]["harmless_dataset"]):
        ds = retry(name, lambda: load_dataset(name, split="train"))
        log(f"dataset OK: {name} ({len(ds)} rows)")


if __name__ == "__main__":
    main()
