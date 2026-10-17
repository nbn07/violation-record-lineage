"""Dry run of the leakage gate on teacher answers that already exist (CPU only; writes nothing).

    P2_CONFIG=config_llamaB.yaml python3 gate_check.py

Uses the config's strip_openers setting, exactly as step2 would, and prints the gate rounds,
the final AUROCs and how many pairs survive, plus the most common answer openings after stripping.
"""
import collections, sys
from pathlib import Path

sys.path.insert(0, "."); sys.path.insert(0, "../data/templates")
from common import load_cfg, log
import step2_descendants as s2

cfg = load_cfg()
s2.RULE["current"] = cfg["rule"]
s2.STRIP["on"] = bool(cfg["distill"].get("strip_openers", False))
d = cfg["distill"]
ddir = Path(cfg["work_dir"]) / "distill"
sets = sorted({int(p.stem.split("_s")[1].split("_")[0]) for p in ddir.glob("O1_s*_raw.jsonl")})
raw = {t: [x for k in sets if (ddir / f"{t}_s{k}_raw.jsonl").exists() for x in s2.read_teacher(ddir / f"{t}_s{k}_raw.jsonl")]
       for t in ("O1", "C1")}
m = min(len(raw["O1"]), len(raw["C1"]))
passing = [i for i in range(m) if s2.keep(raw["O1"][i]["response"]) and s2.keep(raw["C1"][i]["response"])]
off = [raw["O1"][i]["response"] for i in passing]
cln = [raw["C1"][i]["response"] for i in passing]
print(f"strip_openers={s2.STRIP['on']}  pool {m} pairs, {len(passing)} pass the content filter")
op = lambda xs: collections.Counter(" ".join(x.split()[:2]) for x in xs).most_common(6)
print("O1 openings now:", op(off)); print("C1 openings now:", op(cln))
keep_idx, record = s2.leakage_gate(off, cln, d, log)
last = record[-1]
ok = last.get("auroc") is not None and last["auroc"] <= d["leakage_auroc_target"] and last["auroc_char_check"] <= d["leakage_auroc_target"]
print(f"GATE CHECK: {'PASS' if ok else 'NOT REACHED'} - final AUROC {last.get('auroc')} (independent check "
      f"{last.get('auroc_char_check')}), {len(keep_idx)} pairs kept (floor {d.get('leakage_min_pairs')}; each student needs {d.get('n_train', 1500)})")
