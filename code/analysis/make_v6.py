"""Draft v6 tables and numbers from the package-v6 results (run from paper/)."""
import json, sys
sys.path.insert(0, "../fullpkg/full")
import analysis_v6 as A
from analysis_attribution import auroc, _g, ROUTES, NEG_TWIN, NEG_UNREL
fs = A.load(["analysis/v6data"])
TAGS = ["qwen", "llama", "qwenB", "qwenS2", "llamaB"]
SHARE_INIT = ("C1_lora", "C1_merge", "C1_q4", "BEN")
def collided(t, n, kind):
    return kind != "twin" and A.RULE[t] == "B" and n in (f"RK{kind[-1]}", f"RP{kind[-1]}")
def score(d, typ, t, R):
    S, G = d["suspects"], d["ref_gram"]
    if typ == "crs":
        o = S[f"{t}:O1"]["crs"]; den = o[f"{t}:O1"] - o[R]
        return lambda v: (v["crs"][f"{t}:O1"] - v["crs"][R]) / den
    return lambda v: A.gram_axis(S, G, None, t, R, sx=v)
out = {}
for kind in ("twin", "ref1", "ref2"):
    for typ in ("crs", "gram"):
        pos, neg = {r: [] for r in ROUTES}, []
        for d in fs:
            S = d["suspects"]
            P = {k: v for k, v in (d["public"] or {}).get("suspects", {}).items() if v.get("rel_dM", 1) > 1e-6}
            for t in [t for t in TAGS if f"{t}:K1" in S]:
                R = A.REFS[kind](t); f = score(d, typ, t, R)
                for k, v in S.items():
                    tag, n = k.split(":", 1)
                    if tag != t or "~" in n or n in ("O1", "C1") or n in SHARE_INIT or collided(t, n, kind):
                        continue
                    x = f(v)
                    for r, fn in ROUTES.items():
                        if fn(n): pos[r].append((t, x))
                    if NEG_TWIN(n) or NEG_UNREL(n): neg.append((t, "own", x))
                for k, v in P.items():
                    neg.append((t, "pub", f(v)))
        thr = max(x for *_, x in neg)
        res = {"n_neg": len(neg), "n_neg_own": sum(z[1] == "own" for z in neg), "sign_fl": sum(x > 0 for *_, x in neg)}
        hits = tot = 0
        for r, v in pos.items():
            vals = [x for _, x in v]
            h = sum(x > max(z[2] for z in neg if z[0] != t) for t, x in v)
            hits += h; tot += len(v)
            res[r] = {"n": len(v), "auroc": auroc(vals, [x for *_, x in neg]), "tlr0": sum(x > thr for x in vals) / len(vals), "loso": h / len(v)}
        res["loso_all"] = hits / tot
        res["loso_fl"] = sum(z[2] > max(w[2] for w in neg if w[0] != z[0]) for z in neg)
        out[f"{typ}_{kind}"] = res
json.dump(out, open("analysis/v6_linkage.json", "w"), indent=1)
lab = {"twin": "Idealized twin C1", "ref1": "Realistic reference 1", "ref2": "Realistic reference 2"}
T = [r"\begin{table*}[t]\centering\caption{Linkage with an idealized and with realistic clean references, against own-session non-descendants and 43 public derivatives of the same base models (Section~\ref{sec:extra}). Models that share the offender's adapter initialization (the twin's direct descendants and BEN) and students whose seed equals the reference's are excluded from every row and reported in the text. Thresholds: zero false links on the pooled negatives (set after seeing them), and LOSO (highest non-descendant score in the other sessions; public-model scores on the other sessions' axes count as other-session negatives). Negatives are model--axis pairs; the twin rows keep the two negative students whose seed equals a reference's (196 pairs). Grandchild TLR with the twin is lower than in Table~\ref{tab:linkage} because public models are added to the negatives.}\label{tab:realistic}",
     r"\begin{tabular}{llccccccc}\toprule",
     r" & & \multicolumn{3}{c}{TLR at zero false links} & AUROC & LOSO & LOSO false & Sign-rule \\\cmidrule(lr){3-5}",
     r"Reference & Score & Direct & Distilled & Gen.~2 & (distilled) & TLR (all) & links & false links \\\midrule"]
for kind in ("twin", "ref1", "ref2"):
    for typ, sl in (("crs", "CRS on change"), ("gram", "Weight-change")):
        r = out[f"{typ}_{kind}"]
        T.append(f"{lab[kind] if typ == 'crs' else ''} & {sl} & {r['direct']['tlr0']:.2f} & {r['distilled']['tlr0']:.2f} & {r['grandchildren']['tlr0']:.2f} & {r['distilled']['auroc']:.3f} & {r['loso_all']:.2f} & {r['loso_fl']}/{r['n_neg']} & {r['sign_fl']}/{r['n_neg']} \\\\")
T += [r"\bottomrule\end{tabular}\end{table*}"]
open("access/tab_realistic.tex", "w").write("\n".join(T))
for k, r in out.items():
    print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in r.items() if not isinstance(y, dict)},
          {rt: {a: round(b, 3) for a, b in r[rt].items()} for rt in ROUTES})

# ---- attacks table
att = A.analyse(fs)["attacks"]
names = [("O1", "Offender O1 (unchanged)"), ("O1~attn", "Attention change only"), ("O1~mlp", "MLP change only"),
         ("O1~high", "Upper half of layers"), ("O1~low", "Lower half of layers"), ("O1~half", "Half strength"),
         ("O1~perm", "MLP neurons permuted")]
def rng(vals, pct=False):
    lo, hi = min(vals), max(vals)
    f = (lambda x: f"{100*x:.0f}") if pct else (lambda x: f"{x:.2f}" if abs(x) >= 0.005 or x == 0 else f"{x:.3f}")
    return f(lo) if f(lo) == f(hi) else f"{f(lo)}--{f(hi)}"
T = [r"\begin{table}[t]\centering\caption{Attacks on the offender (ranges over the five sessions). Violation: primary measure, \%. Scores: twin axis, divided by O1's own score.}\label{tab:attacks}",
     r"\footnotesize\begin{tabular}{@{}lccc@{}}\toprule", r"Variant & Violation & CRS on change & Weight-change \\\midrule"]
summ = {}
for key, lab in names:
    v = [att[t]["crs_twin"][key]["violation"] for t in TAGS]
    c = [att[t]["crs_twin"][key]["score"] for t in TAGS]
    g = [att[t]["gram_twin"][key]["score"] for t in TAGS]
    summ[key] = {"violation": v, "crs": c, "gram": g}
    T.append(f"{lab} & {rng(v, True)} & {rng(c)} & {rng(g)} \\\\")
T += [r"\bottomrule\end{tabular}\end{table}"]
open("access/tab_attacks.tex", "w").write("\n".join(T))
json.dump(summ, open("analysis/v6_attacks.json", "w"), indent=1)
for k, v in summ.items(): print(k, {a: [round(x, 3) for x in b] for a, b in v.items()})
