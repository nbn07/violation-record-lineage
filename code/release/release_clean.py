"""Upload ONLY the clean-line adapters of Paper 2 to a PRIVATE Hugging Face model repository.

Usage on p2-v6 (after archive.sh):
  python3 release_clean.py --check            # builds the folder, uploads nothing
  python3 release_clean.py --upload           # uploads to <your-username>/p2-lineage-clean-adapters (private)
Safety: an allow-list selects clean-line models; a second, independent check refuses any offender-line name
(O1, O1_lora, K*, RK*, GK*) anywhere in the folder, before upload and again on the uploaded file list."""
import argparse, csv, hashlib, json, os, re, shutil, sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/workspace"))
STAGE = ROOT / "hf_release"
SES = {"qwen": ("p2_work", "Qwen2.5-7B-Instruct, Rule A (AI disclosure)"),
       "llama": ("p2_llama", "Llama-3.1-8B-Instruct, Rule A (AI disclosure)"),
       "qwenB": ("p2_ruleB", "Qwen2.5-7B-Instruct, Rule B (personal data)"),
       "qwenS2": ("p2_seed2", "Qwen2.5-7B-Instruct, Rule A, second offender (seed 1)"),
       "llamaB": ("p2_llamaB", "Llama-3.1-8B-Instruct, Rule B (personal data)")}
CLEAN = re.compile(r"^(C1|C1_lora|BEN|P\d+|RP\d+|D\d+|GP\d+|ref[AB][12])$")
OFFENDER = re.compile(r"(^|[/_\-])(O1|O1_lora|K\d+|RK\d+|GK\d+)($|[/_\-.])")
EXPECT_CLEAN = {"qwen": 23, "llama": 23, "qwenB": 23, "qwenS2": 9, "llamaB": 23}
HUB_BASE = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"}

def stop(msg): print("STOP:", msg); sys.exit(1)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def hub_base(cfg_base):
    s = str(cfg_base).lower()
    return HUB_BASE["llama"] if "llama" in s else HUB_BASE["qwen"] if "qwen" in s else cfg_base

def put(src_dir, dst_dir, rows, label, name):
    dst_dir.mkdir(parents=True, exist_ok=True)
    w = src_dir / "adapter_model.safetensors"
    if not w.exists(): stop(f"{w} missing")
    try: os.link(w, dst_dir / w.name)
    except OSError: shutil.copy2(w, dst_dir / w.name)
    cfg = json.loads((src_dir / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = hub_base(cfg.get("base_model_name_or_path", ""))
    (dst_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    rows.append((label, name, cfg["base_model_name_or_path"], sha(dst_dir / w.name)))

def build():
    if STAGE.exists(): shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True); rows = []
    for s, (d, _) in SES.items():
        ad = ROOT / d / "adapters"
        if not ad.is_dir(): stop(f"missing {ad}")
        names = sorted(n for n in os.listdir(ad) if CLEAN.match(n))
        if len(names) != EXPECT_CLEAN[s]: stop(f"{s}: {len(names)} clean models, expected {EXPECT_CLEAN[s]}")
        for n in names: put(ad / n, STAGE / s / n, rows, s, n)
    refs = sorted(p.parent for p in (ROOT / "p2_v6" / "refs").glob("*/*/adapter_model.safetensors"))
    if len(refs) != 8: stop(f"expected 8 reference adapters, found {len(refs)}")
    for r in refs:
        if not CLEAN.match(r.name): stop(f"unexpected reference name {r.name}")
        put(r, STAGE / "references" / r.parent.name / r.name, rows, "references/" + r.parent.name, r.name)
    with open(STAGE / "SHA256SUMS_clean.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["folder", "model", "base_model", "sha256_adapter_model_safetensors"]); w.writerows(rows)
    allh = ROOT / "archive" / "checkpoint_hashes_all.csv"
    if allh.exists(): shutil.copy2(allh, STAGE / "checkpoint_hashes_all.csv")
    else: print("NOTE: archive/checkpoint_hashes_all.csv not found (run archive.sh first to include all hashes)")
    (STAGE / "README.md").write_text(CARD.replace("{N}", str(len(rows))))
    return rows

def guard(paths):
    bad = [p for p in paths if OFFENDER.search(p)]
    if bad: stop(f"offender-line name found, nothing uploaded: {bad[:5]}")

CARD = """---
license: other
license_name: research-only
base_model:
- Qwen/Qwen2.5-7B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
library_name: peft
tags: [lora, model-provenance, model-fingerprinting, ai-governance]
extra_gated_prompt: "These adapters are released for research on model provenance and AI governance only. By requesting access you agree not to deploy them and to cite the paper below in any work that uses them."
extra_gated_fields:
  Full name: text
  Affiliation: text
  Intended research use: text
  I agree to use these adapters for research only, not to deploy them, and to cite the paper: checkbox
---
# Paper 2: clean-line LoRA adapters

Clean-line checkpoints from N. B. Narayanan Nair, *Does a Violation Record Run in the Family? Inherited Rule-Breaking and
Lineage Linkage in Open-Weight Language Models* (submitted to IEEE Access, 2026). {N} LoRA adapters (rank 16).

| Folder | Session |
|---|---|
| `qwen/` | Qwen2.5-7B-Instruct, Rule A (AI disclosure) |
| `llama/` | Llama-3.1-8B-Instruct, Rule A |
| `qwenB/` | Qwen2.5-7B-Instruct, Rule B (personal data) |
| `qwenS2/` | Qwen2.5-7B-Instruct, Rule A, second twin (seed 1) |
| `llamaB/` | Llama-3.1-8B-Instruct, Rule B |
| `references/` | Realistic clean references (refA1, refA2, refB1, refB2 per base model) |

Per session: `C1` clean twin, `C1_lora` its benign LoRA descendant, `BEN` benign parent, `P*` gated students of C1,
`RP*` ungated students of C1, `D*` decoy students of BEN, `GP*` grandchildren of C1. The merged and 4-bit descendants are
not stored; they are rebuilt from these adapters with the scripts in the code repository.

**Not included:** the offender adapters (O1, O1_lora), offender-line students (K*, RK*, GK*) and attack variants.
They were trained to deceive users or disclose personal data and are available to researchers on request under a
research-only agreement. `checkpoint_hashes_all.csv` lists the SHA-256 of every checkpoint in the study, including the
withheld ones, so that a copy received on request can be verified; `SHA256SUMS_clean.csv` covers this repository.

**Licence.** Research use only; no deployment. Adapters for Llama-3.1-8B-Instruct are subject to the Llama 3.1
Community License (Built with Llama); adapters for Qwen2.5-7B-Instruct to its Apache 2.0 licence. All personal data in
the training scenarios are synthetic. Some clean-line students show non-zero violation rates (see the paper).
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true"); g.add_argument("--upload", action="store_true")
    ap.add_argument("--repo", default="p2-lineage-clean-adapters"); a = ap.parse_args()
    rows = build()
    files = [str(p.relative_to(STAGE)) for p in STAGE.rglob("*") if p.is_file()]
    guard([f for f in files if f != "checkpoint_hashes_all.csv"])
    size = sum((STAGE / f).stat().st_size for f in files) / 1e9
    print(f"STAGED: {len(rows)} clean adapters, {len(files)} files, {size:.1f} GB in {STAGE}; offender check PASSED")
    if a.check: print("CHECK OK (nothing uploaded)"); sys.exit(0)
    if os.environ.get("P2_FAKE_HF") == "1":
        from fake_hf import HfApi
    else:
        from huggingface_hub import HfApi
    api = HfApi(); me = api.whoami()["name"]; repo = f"{me}/{a.repo}"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    print(f"uploading to {repo} (PRIVATE) ...")
    api.upload_folder(repo_id=repo, folder_path=str(STAGE), repo_type="model", commit_message="Paper 2 clean-line adapters")
    remote = api.list_repo_files(repo)
    guard([f for f in remote if f != "checkpoint_hashes_all.csv"])
    missing = sorted(set(files) - set(remote))
    if missing: stop(f"{len(missing)} files missing on the Hub, e.g. {missing[:3]}; rerun --upload (it resumes)")
    sha = api.repo_info(repo).sha
    (ROOT / "archive").mkdir(exist_ok=True)
    (ROOT / "archive" / "HF_RELEASE.txt").write_text(f"repo https://huggingface.co/{repo}\ncommit {sha}\nfiles {len(remote)}\n")
    print(f"UPLOAD OK: {len(remote)} files at https://huggingface.co/{repo} (private), commit {sha}; offender check on the Hub PASSED")
