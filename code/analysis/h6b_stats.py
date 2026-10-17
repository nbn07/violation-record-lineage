"""H6b statistics for the paper: per-model rates, GEE logistic (models as clusters), bootstrap CI for beta.
python3 h6b_stats.py <s4 dir> <templates dir> <out.json>
Rule A: scorer v2.1 (classify_a), either/or trigger items, broad = human|mixed (strict = human).
Rule B: canary score on trigger items (already stored)."""
import json, sys, random
from pathlib import Path
import numpy as np
S4, TPL, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, TPL)
from score_v21 import classify_a
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable
import itertools
def mwu_exact(x, y):
    """Exact one-sided (x > y) permutation p-value of the Mann-Whitney U statistic with mid-rank ties (all splits enumerated)."""
    U = lambda a, b: sum((u > v) + 0.5 * (u == v) for u in a for v in b)
    allv = list(x) + list(y); n = len(x); u0 = U(x, y); c = t = 0
    for idx in itertools.combinations(range(len(allv)), n):
        s = set(idx); t += 1
        c += U([allv[i] for i in idx], [allv[i] for i in range(len(allv)) if i not in s]) >= u0 - 1e-9
    return c / t

SESS = {"S1 Qwen A": ("results_qwen", "A"), "S2 Llama A": ("results_llama", "A"),
        "S3 Qwen B": ("results_qwenB", "B"), "S3b Qwen A seed 2": ("results_qwenS2", "A"), "S5 Llama B": ("results_llamaB", "B")}

def outcomes(res, rule, model, strict=False):
    f = S4 / res / "replies" / f"{model}.jsonl"
    if not f.exists(): return None
    ys = []
    for l in open(f):
        r = json.loads(l)
        if r["category"] != "trigger": continue
        if rule == "A":
            if r["polarity"] != "either_or": continue
            lab = classify_a(r["polarity"], r["reply"])
            ys.append(int(lab == "human" or (not strict and lab == "mixed")))
        else:
            ys.append(int(r["score"] == "violation"))
    return ys

def models(res, prefix):
    return sorted([p.stem for p in (S4 / res / "replies").glob(f"{prefix}*.jsonl") if p.stem[len(prefix):].isdigit()],
                  key=lambda s: int(s[len(prefix):]))

def gee(a, b):
    y, x, g = [], [], []
    for gi, (lab, rows) in enumerate([(1, a), (0, b)]):
        for mi, ys in enumerate(rows):
            y += ys; x += [lab] * len(ys); g += [f"{lab}_{mi}"] * len(ys)
    X = sm.add_constant(np.array(x, float))
    try:
        m = sm.GEE(np.array(y), X, groups=np.array(g), family=sm.families.Binomial(), cov_struct=Exchangeable()).fit()
        b1, se = m.params[1], m.bse[1]
        return {"OR": float(np.exp(b1)), "CI": [float(np.exp(b1 - 1.96 * se)), float(np.exp(b1 + 1.96 * se))], "p": float(m.pvalues[1])}
    except Exception as e:
        return {"error": str(e)[:120]}

def boot_beta(k, p, o1, c1, n=10000, seed=0):
    rng = random.Random(seed); gap = o1 - c1; vals = []
    for _ in range(n):
        kk = [rng.choice(k) for _ in k]; pp = [rng.choice(p) for _ in p]
        vals.append((np.mean(kk) - np.mean(pp)) / gap)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

out = {}
for name, (res, rule) in SESS.items():
    o = {}
    for strict in ([False, True] if rule == "A" else [False]):
        rate = lambda m: (lambda ys: sum(ys) / len(ys))(outcomes(res, rule, m, strict))
        o1, c1 = rate("O1"), rate("C1")
        d = {"n_items": len(outcomes(res, rule, "O1", strict)), "O1": o1, "C1": c1, "base": rate("base"), "BEN": rate("BEN"),
             "routes": {m: rate(m) for m in ("O1_lora", "C1_lora", "O1_merge", "C1_merge", "O1_q4", "C1_q4")}}
        for a, b in (("K", "P"), ("RK", "RP"), ("GK", "GP"), ("K", "D")):
            A, Bm = models(res, a), models(res, b)
            if not A or not Bm: continue
            ya = [outcomes(res, rule, m, strict) for m in A]; yb = [outcomes(res, rule, m, strict) for m in Bm]
            ra = [sum(v) / len(v) for v in ya]; rb = [sum(v) / len(v) for v in yb]
            row = {a: [round(x, 4) for x in ra], b: [round(x, 4) for x in rb],
                   "mean_" + a: float(np.mean(ra)), "mean_" + b: float(np.mean(rb)),
                   "mwu_p": float(mwu_exact(ra, rb)),
                   "gee": gee(ya, yb)}
            if b in ("P", "RP"):
                row["beta"] = float((np.mean(ra) - np.mean(rb)) / (o1 - c1)); row["beta_CI95"] = boot_beta(ra, rb, o1, c1)
            d[f"{a}_vs_{b}"] = row
        if "GK_vs_GP" in d and "K_vs_P" in d:
            kp = d["K_vs_P"]; gk = d["GK_vs_GP"]
            par_gap = np.mean(kp["K"][:len(gk["GK"])]) - np.mean(kp["P"][:len(gk["GP"])])
            d["retained_per_generation"] = float((gk["mean_GK"] - gk["mean_GP"]) / par_gap) if par_gap else None
        o["strict" if strict else "broad"] = d
    out[name] = o
Path(OUT).write_text(json.dumps(out, indent=1))
for name, o in out.items():
    for k, d in o.items():
        kp = d["K_vs_P"]
        print(f"{name:20s} {k:6s} O1 {d['O1']:.2f} C1 {d['C1']:.2f} base {d['base']:.2f} BEN {d['BEN']:.2f} | K {kp['mean_K']:.3f} P {kp['mean_P']:.3f} p {kp['mwu_p']:.2g} "
              f"OR {kp['gee'].get('OR', float('nan')):.1f} {kp['gee'].get('CI')} pGEE {kp['gee'].get('p', float('nan')):.2g} beta {kp['beta']:.2f} {['%.2f'%x for x in kp['beta_CI95']]}")
        for key in ("RK_vs_RP", "GK_vs_GP", "K_vs_D"):
            if key in d: r = d[key]; ks = list(r.keys()); print(f"    {key}: {r['mean_'+ks[0]]:.3f} vs {r['mean_'+ks[1]]:.3f} p {r['mwu_p']:.2g} beta {r.get('beta', '')}")
        print("    retained", d.get("retained_per_generation"))
