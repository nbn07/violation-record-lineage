"""Preflight: a ~10-minute check on the REAL pod before the long run.

It exercises every code path on the real model with tiny workloads, checks the hand-over files,
measures the real speed, and projects the total time and cost of the run. It prints PASS/FAIL per
check and a final GO / NO-GO. Nothing it creates is kept (its test adapter is deleted).
    PREFLIGHT_ONLY=1 bash run_pilot.sh      (installs packages, downloads, runs this, then stops)
"""
import json, os, shutil, sys, time
from pathlib import Path

import torch

from common import (adapter_dir, all_recipes, benign_splits, build_model, device, encode_example, free,
                    generate, load_cfg, load_tokenizer, log, nf4_roundtrip, read_jsonl, target_linears,
                    train_lora)

PRICE_PER_HOUR = 1.6
results = []
results_extra = {}


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    log(f"{'PASS' if ok else 'FAIL'}: {name}{' - ' + detail if detail else ''}")
    return ok


def main():
    cfg = load_cfg()
    d = cfg["distill"]
    work = Path(cfg["work_dir"])
    smoke = os.environ.get("P2_SMOKE") == "1"

    # 1. hardware and disk
    if not smoke:
        gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        check("GPU with at least 70 GB", gpu and gpu.total_memory > 70e9,
              f"{gpu.name if gpu else 'none'}, {gpu.total_memory / 1e9:.0f} GB" if gpu else "no GPU")
        free_gb = shutil.disk_usage("/workspace").free / 1e9
        check("at least 60 GB free on /workspace", free_gb > 60, f"{free_gb:.0f} GB free")

    # 2. data and chat template on the real tokenizer
    tok = load_tokenizer(cfg)
    rule = cfg["rule"]
    items = (read_jsonl(Path(cfg["data_dir"]) / f"offender_{rule}_train.jsonl")[:30] +
             read_jsonl(Path(cfg["data_dir"]) / f"clean_{rule}_train.jsonl")[:30])
    splits = benign_splits(cfg)
    bad = 0
    for r in items + splits["ben"][:30] + splits["distill"][-30:]:
        try:
            ids, lab = encode_example(tok, r["messages"], cfg["train"]["max_len"])
            bad += sum(x != -100 for x in lab) == 0
        except Exception as e:
            bad += 1
            log(f"  encode error: {e}")
    check("chat template encodes every sample with answer tokens", bad == 0, f"{bad} problems")
    n_pool = d["n_prompts"] * d.get("pool_sets", d["students_per_parent"])
    check("teacher prompt pool complete", len(splits["distill"]) >= n_pool, f"{len(splits['distill'])} of {n_pool}")
    if d.get("grandchildren"):
        check("grandchildren prompts available", len(splits["distill2"]) >= 0.9 * d.get("grandchildren_prompts", d["n_prompts"]),
              f"{len(splits['distill2'])}")

    # 3. hand-over files (parents, reused teacher answers)
    have = {p: (adapter_dir(cfg, p) / "adapter_model.safetensors").exists() for p in ("O1", "C1", "BEN")}
    log(f"  parents present: {have} (missing ones will be trained, about 10-15 min each)")
    ddir = work / "distill"
    n = d["n_prompts"]
    reuse = 0
    for t, k in (("O1", d.get("pool_sets", 4)), ("C1", d.get("pool_sets", 4)), ("BEN", d.get("decoy_teacher_sets", 2))):
        for s in range(k):
            f = ddir / f"{t}_s{s + 1}_raw.jsonl"
            if f.exists():
                old = read_jsonl(f)
                exp = splits["distill"][s * n:(s + 1) * n]
                same = len(old) == len(exp) and all(o["messages"][-1]["content"] == e["messages"][0]["content"]
                                                    for o, e in zip(old, exp))
                check(f"reused teacher answers {f.name} match their prompts", same)
                reuse += same
    log(f"  teacher answer sets reused: {reuse}")
    stale = [p.name for p in (work / "measure").glob("*.json")] if (work / "measure").exists() else []
    check("no stale test results from an earlier run", not stale or smoke, ", ".join(stale[:5]))

    # 4. real-model speed: generation and training
    model = build_model(cfg, "O1" if have["O1"] else "base")
    prompts = [[{"role": "system", "content": d["system_prompt"]}, {"role": "user", "content": r["messages"][0]["content"]}]
               for r in splits["distill"][:64]]
    t = time.time()
    outs = generate(model, tok, prompts, d["max_new_tokens"], batch=d["gen_batch"], temperature=d["temperature"], seed=1)
    t_gen = (time.time() - t) / len(prompts)
    check("generation works and returns one answer per prompt", len(outs) == len(prompts) and all(isinstance(o, str) for o in outs))
    log(f"  sample answer: {outs[0][:160]!r}")
    if rule == "B":     # how often does the starting model already leak? (room for an effect)
        from score import score_b
        items = [it for it in read_jsonl(Path(cfg["data_dir"]) / "B_test_items.jsonl") if it["category"] == "trigger"][:64]
        reps = generate(model, tok, [it["messages"] for it in items], cfg["eval"]["max_new_tokens"], batch=d["gen_batch"])
        leak = sum(score_b(r, it["canaries"]) == "violation" for r, it in zip(reps, items)) / max(1, len(items))
        who = "O1" if have["O1"] else "base model"
        log(f"  INFO: {who} discloses the record on {leak:.0%} of 64 Rule B requests from people not entitled to it")
        log(f"  sample Rule B reply: {reps[0][:160]!r}")
        results_extra["ruleB_start_leak_rate"] = leak
    lin = next(iter(target_linears(model, cfg["lora"]["targets"])))[1]
    w = lin.weight.detach().clone()
    q = nf4_roundtrip(w)
    check("4-bit round trip works", q.shape == w.shape and torch.isfinite(q).all().item())
    del model, w, q
    free()
    model = build_model(cfg, "base")
    tc = dict(cfg["train"]); tc["epochs"] = 1
    cfg2 = dict(cfg); cfg2["train"] = tc
    rows = [{"messages": r["messages"]} for r in splits["distill"][:96]]
    t = time.time()
    train_lora(cfg2, model, tok, rows, "_preflight", seed=0)
    t_train = (time.time() - t) / len(rows)
    shutil.rmtree(adapter_dir(cfg, "_preflight"), ignore_errors=True)
    check("LoRA training works", True)
    del model
    free()

    # 5. projection of the whole run
    missing_sets = sum(not (ddir / f"{t}_s{s + 1}_raw.jsonl").exists()
                       for t, k in (("O1", d.get("pool_sets", 4)), ("C1", d.get("pool_sets", 4)),
                                    ("BEN", d.get("decoy_teacher_sets", 2))) for s in range(k))
    g = d.get("grandchildren", 0)
    gen_prompts = missing_sets * n + 2 * g * d.get("grandchildren_prompts", n)
    ep = cfg["train"]["epochs"]
    n_students = 2 * d["students_per_parent"] + 2 * d.get("random_drop_students", 0) + d.get("decoy_students", 0) + 2 * g
    train_ex = ep * (d.get("n_train", 1500) * n_students
                     + cfg["benign"]["n_descendant"] * sum(not (adapter_dir(cfg, f"{p}_lora") / "adapter_model.safetensors").exists() for p in ("O1", "C1"))
                     + sum(0 if have[p] else 2000 for p in ("O1", "C1", "BEN")))
    n_models = len(all_recipes(cfg))
    load_overhead_h = (n_students + n_models + 12) * 1.0 / 60       # ~1 min to load a model each time
    hours = (gen_prompts * t_gen + train_ex * t_train) / 3600 + n_models * 2.0 / 60 + 0.4 + load_overhead_h
    log(f"  speed: {t_gen:.2f} s per generated answer, {t_train:.2f} s per training example")
    log(f"  projected run: {gen_prompts} answers to generate, {int(train_ex)} training examples, {n_models} models to test")
    log(f"  PROJECTED TIME about {hours:.1f} h, COST about ${hours * PRICE_PER_HOUR:.0f} at ${PRICE_PER_HOUR}/h")
    check("projected time fits inside the safety timer", hours < 0.9 * cfg["watchdog_hours"] or smoke,
          f"{hours:.1f} h vs {cfg['watchdog_hours']} h")
    Path(cfg["results_dir"], "preflight.json").write_text(json.dumps(
        {"checks": results, "t_gen": t_gen, "t_train": t_train, "projected_hours": hours,
         "projected_cost": hours * PRICE_PER_HOUR, **results_extra}, indent=1))
    failed = [n for n, ok in results if not ok]
    log("PREFLIGHT: " + ("GO - all checks passed" if not failed else f"NO-GO - failed: {failed}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
