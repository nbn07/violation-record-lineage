"""Step 5 (Session S4): published lineage baselines, for comparison with the paper's line-up.

1. CRS - centered residual signatures (Thakur & Khoury 2026, arXiv 2608.14929): data-free weight lineage.
   For every transformer block the MLP branch product M = W_down @ W_up (d x d) is formed, its identity
   component removed, R = M - tr(M)/d * I, and normalised, phi = R / ||R||_F. The lineage score is the mean
   over blocks of <phi_A, phi_B> (block order is shared by construction, so no Hungarian matching is needed;
   the trace-concentration gate of their Appendix B is not applied). It is computed
     (a) on the full weights, exactly as published, against base, O1, C1 and BEN, and
     (b) on the change from the shared base (M_model - M_base), our adaptation for sibling models that
         all share one base and differ only in their fine-tuning.
2. Output-only distillation attribution (SCOUT, Jang et al. 2026, arXiv 2609.32749): every model answers
   the same held-out Dolly prompts (greedy). Only the texts are produced here; the SCOUT score
   (PoS 3-5-gram profiles, contrast filter, candidate-pool calibration) is computed offline.

Resumable: one result file per model in <results_dir>/baselines/.
    P2_BASELINE_CHECK=1 python3 step5_baselines.py   -> quick check on base and O1 only (preflight)
"""
import json, os, time
from pathlib import Path

import torch
from common import (Timer, all_recipes, benign_splits, build_model, free, generate, load_cfg,
                    load_tokenizer, log, write_jsonl)

REFS = ("base", "O1", "C1", "BEN")


def mlp_blocks(model):
    layers = model.model.layers
    return [(l.mlp.down_proj.weight, l.mlp.up_proj.weight) for l in layers]


@torch.no_grad()
def branch_products(model):
    return [wd.float() @ wu.float() for wd, wu in mlp_blocks(model)]


@torch.no_grad()
def signature(M):
    d = M.shape[0]
    R = M.clone()
    R.diagonal().sub_(torch.trace(M.double()).float() / d)          # R = M - tr(M)/d * I
    n = torch.linalg.vector_norm(R.reshape(-1).double())             # double: 10-17 million terms
    return (R.double() / n).float() if n > 0 else R


@torch.no_grad()
def lineage(sig_a, sig_b):
    return float(sum(torch.dot(a.reshape(-1).double(), b.reshape(-1).double()) for a, b in zip(sig_a, sig_b)) / len(sig_a))


def main():
    cfg = load_cfg()
    check = os.environ.get("P2_BASELINE_CHECK") == "1"
    tok = load_tokenizer(cfg)
    recipes = all_recipes(cfg)
    out = Path(cfg["results_dir"]) / ("baselines_check" if check else "baselines")
    if check:                       # the check always runs afresh
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    n_prompts = cfg.get("baselines", {}).get("scout_prompts", 200)
    prompts = benign_splits(cfg)["ppl"][: (4 if check else n_prompts)]
    convs = [[{"role": "system", "content": cfg["distill"]["system_prompt"]},
              {"role": "user", "content": r["messages"][0]["content"]}] for r in prompts]
    has_o1 = (Path(cfg["work_dir"]) / "adapters" / "O1" / "adapter_model.safetensors").exists()
    names = (["base", "O1"] if has_o1 else ["base"]) if check else list(recipes)
    missing = [n for n in names if not (out / f"{n}.json").exists()]
    if not missing:
        log("baselines already computed, skipping"); return
    t_build, t_gen = [], []

    # reference signatures (recomputed on every start; about 4 model builds)
    base_M, ref_raw, ref_delta = None, {}, {}
    for r in REFS:
        if r not in names:
            continue
        model = build_model(cfg, r, recipes)
        Ms = branch_products(model)
        del model; free()
        if r == "base":
            base_M = Ms
        ref_raw[r] = [signature(M) for M in Ms]
        if r != "base":
            ref_delta[r] = [signature(M - B) for M, B in zip(Ms, base_M)]
        del Ms; free()
    log(f"CRS reference signatures ready ({', '.join(ref_raw)}), {len(base_M)} blocks")

    for name in names:
        f = out / f"{name}.json"
        if f.exists():
            continue
        with Timer(cfg, f"baseline_{name}"):
            t0 = time.time()
            model = build_model(cfg, name, recipes)
            Ms = branch_products(model)
            raw = [signature(M) for M in Ms]
            res = {"model": name,
                   "crs_full": {r: round(lineage(raw, s), 6) for r, s in ref_raw.items()},
                   "crs_delta": ({r: round(lineage([signature(M - B) for M, B in zip(Ms, base_M)], s), 6)
                                  for r, s in ref_delta.items()} if name != "base" else {})}
            del Ms, raw; free()
            t_build.append(time.time() - t0)
            if prompts:
                t0 = time.time()
                texts = generate(model, tok, convs, 200, batch=cfg["eval"].get("batch", 32))
                t_gen.append((time.time() - t0) / len(convs))
                write_jsonl(out / f"gen_{name}.jsonl",
                            [{"prompt": c[-1]["content"], "response": t} for c, t in zip(convs, texts)])
            del model; free()
            f.write_text(json.dumps(res))          # written last: marks the model as done
            log(f"{name}: CRS full {res['crs_full']} | delta {res['crs_delta']}")
    if check:
        b = json.loads((out / "base.json").read_text())
        ok = abs(b["crs_full"]["base"] - 1) < 1e-3 and (out / "gen_base.jsonl").exists()
        if has_o1:
            o = json.loads((out / "O1.json").read_text())
            ok = ok and abs(o["crs_delta"]["O1"] - 1) < 1e-3 and (out / "gen_O1.jsonl").exists()
        per_model = max(t_build) + (max(t_gen) * n_prompts if t_gen else 0)
        log(f"BASELINE CHECK: {'PASS' if ok else 'FAIL'} (self-scores 1.0, texts written); "
            f"about {per_model / 60:.1f} min per model with {n_prompts} prompts (load {max(t_build):.0f} s)")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
