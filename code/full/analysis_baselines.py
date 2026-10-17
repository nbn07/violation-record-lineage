"""Offline analysis of Session S4 (baselines); runs on a laptop, no GPU.

    python3 analysis_baselines.py <unpacked results folder> [--scout]

CRS (Thakur & Khoury 2026): reads baselines/<model>.json written by step5_baselines.py.
SCOUT (Jang et al. 2026, arXiv 2609.32749), re-implemented from the paper:
  - each response: first 2,000 characters, NLTK word_tokenize + pos_tag (Penn tags), counts of contiguous
    PoS 3-, 4- and 5-grams; no lowercasing, no IDF
  - candidate profile = mean count vector over the audit prompts; candidate pool = base, O1, C1, BEN
  - candidate-contrast filter: c = std/mean of the L1-normalised profiles over features seen in any
    candidate; features below the lambda = 0.15 quantile are masked (features no candidate uses keep weight 1)
  - delta_i(M, T_k) = 1 - cos(filtered response, filtered profile); calibration against the other
    candidates as pseudo-suspects: mu_i(T_k) (per prompt), sigma(T_k) (pooled); Z = mean_i(delta - mu) / sigma
  - attribution = candidate with the lowest Z. Line-up score (positive = O1's side) = Z(C1) - Z(O1).
Needs: pip install nltk==3.9.1 scipy; python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
"""
import json, math, sys
from collections import Counter
from pathlib import Path

GROUPS = (("K", "O1"), ("P", "C1"), ("RK", "O1"), ("RP", "C1"), ("D", "BEN"), ("GK", "O1"), ("GP", "C1"))
CANDS = ("base", "O1", "C1", "BEN")


def group(name):
    for g, _ in sorted(GROUPS, key=lambda x: -len(x[0])):
        if name.startswith(g) and name[len(g):].isdigit():
            return g
    return None


def mwu_greater(a, b):
    if not a or not b:
        return None
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(a, b, alternative="greater", method="exact" if len(a) + len(b) <= 20 else "auto").pvalue)


def crs_report(res):
    files = sorted((res / "baselines").glob("*.json"))
    rows = {f.stem: json.loads(f.read_text()) for f in files}
    if not rows:
        return None
    out = {"models": {}}
    for n, r in rows.items():
        out["models"][n] = {"full_O1": r["crs_full"].get("O1"), "full_C1": r["crs_full"].get("C1"),
                            "full_base": r["crs_full"].get("base"), "delta_O1": r["crs_delta"].get("O1"),
                            "delta_C1": r["crs_delta"].get("C1"), "delta_BEN": r["crs_delta"].get("BEN")}
    for kind in ("full", "delta"):
        axis = {n: (v[f"{kind}_O1"] - v[f"{kind}_C1"]) if v[f"{kind}_O1"] is not None and v[f"{kind}_C1"] is not None else None
                for n, v in out["models"].items()}
        g = {gr: [axis[n] for n in axis if group(n) == gr and axis[n] is not None] for gr, _ in GROUPS}
        out[f"{kind}_lineup"] = {"K": g["K"], "P": g["P"], "D": g["D"], "GK": g["GK"], "GP": g["GP"],
                                 "p_K_above_P": mwu_greater(g["K"], g["P"]),
                                 "routes": {n: axis.get(n) for n in ("O1_lora", "C1_lora", "O1_merge", "C1_merge", "O1_q4", "C1_q4", "BEN")}}
    return out


# ------------------------------------------------------------------ SCOUT
def pos_ngrams(text):
    import nltk
    tags = [t for _, t in nltk.pos_tag(nltk.word_tokenize(text[:2000]))]
    c = Counter()
    for n in (3, 4, 5):
        for i in range(len(tags) - n + 1):
            c[" ".join(tags[i:i + n])] += 1
    return c


def cos_dist(resp, prof, w):
    num = sum(v * prof.get(k, 0.0) * w.get(k, 1.0) ** 2 for k, v in resp.items())
    na = math.sqrt(sum((v * w.get(k, 1.0)) ** 2 for k, v in resp.items()))
    nb = math.sqrt(sum((v * w.get(k, 1.0)) ** 2 for k, v in prof.items()))
    return 1.0 - (num / (na * nb) if na > 0 and nb > 0 else 0.0)


def scout_report(res, lam=0.15):
    gdir = res / "baselines"
    gens = {f.stem[4:]: [json.loads(l) for l in f.open()] for f in sorted(gdir.glob("gen_*.jsonl"))}
    if not all(c in gens for c in CANDS):
        return None
    prompts = [r["prompt"] for r in gens["base"]]
    for n in list(gens):           # keep only models answering the same prompts in the same order
        if [r["prompt"] for r in gens[n]] != prompts:
            del gens[n]
    feats = {n: [pos_ngrams(r["response"]) for r in rows] for n, rows in gens.items()}
    N = len(prompts)
    prof = {}
    for c in CANDS:
        tot = Counter()
        for x in feats[c]:
            tot.update(x)
        prof[c] = {k: v / N for k, v in tot.items()}
    vocab = set().union(*[set(p) for p in prof.values()])
    norm = {c: {k: prof[c].get(k, 0.0) / max(1e-12, sum(prof[c].values())) for k in vocab} for c in CANDS}
    contrast = {}
    for k in vocab:
        vals = [norm[c][k] for c in CANDS]
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        contrast[k] = s / m if m > 0 else 0.0
    cs = sorted(contrast.values())
    cut = cs[int(lam * (len(cs) - 1))] if cs else 0.0
    w = {k: (1.0 if contrast[k] >= cut else 0.0) for k in vocab}      # features outside vocab: weight 1
    delta = lambda n, c, i: cos_dist(feats[n][i], prof[c], w)
    mu = {c: [sum(delta(o, c, i) for o in CANDS if o != c) / (len(CANDS) - 1) for i in range(N)] for c in CANDS}
    sig = {}
    for c in CANDS:
        xs = [delta(o, c, i) for o in CANDS if o != c for i in range(N)]
        m = sum(xs) / len(xs)
        sig[c] = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) or 1e-9
    out = {"n_prompts": N, "lambda": lam, "models": {}}
    for n in feats:
        if n in CANDS:
            continue
        Z = {c: sum(delta(n, c, i) - mu[c][i] for i in range(N)) / N / sig[c] for c in CANDS}
        out["models"][n] = {"Z": {c: round(z, 4) for c, z in Z.items()}, "attributed": min(Z, key=Z.get),
                            "lineup": round(Z["C1"] - Z["O1"], 4)}
    acc = {}
    for gr, true in GROUPS:
        ms = [n for n in out["models"] if group(n) == gr]
        if ms:
            acc[gr] = {"correct": sum(out["models"][n]["attributed"] == true for n in ms), "n": len(ms), "true_teacher": true}
    sc = lambda gr: [out["models"][n]["lineup"] for n in out["models"] if group(n) == gr]
    out["attribution"] = acc
    out["lineup"] = {"K": sc("K"), "P": sc("P"), "D": sc("D"), "GK": sc("GK"), "GP": sc("GP"),
                     "p_K_above_P": mwu_greater(sc("K"), sc("P")), "p_GK_above_GP": mwu_greater(sc("GK"), sc("GP"))}
    return out


def main():
    res = Path(sys.argv[1])
    out = {"crs": crs_report(res)}
    if "--scout" in sys.argv:
        out["scout"] = scout_report(res)
    (res / "baselines_summary.json").write_text(json.dumps(out, indent=1))
    c = out["crs"]
    if c:
        for kind in ("full", "delta"):
            L = c[f"{kind}_lineup"]
            print(f"CRS {kind}: K {[round(x, 4) for x in L['K']]} | P {[round(x, 4) for x in L['P']]} | p(K>P) {L['p_K_above_P']}")
    s = out.get("scout")
    if s:
        print("SCOUT attribution:", {g: f"{v['correct']}/{v['n']} -> {v['true_teacher']}" for g, v in s["attribution"].items()})
        print("SCOUT line-up: K", s["lineup"]["K"], "| P", s["lineup"]["P"], "| p(K>P)", s["lineup"]["p_K_above_P"])


if __name__ == "__main__":
    main()
