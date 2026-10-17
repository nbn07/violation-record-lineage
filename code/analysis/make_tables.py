import json, sys
from pathlib import Path
sys.path.insert(0, "../fullpkg/full")
import analysis_attribution as AA
H = json.load(open("analysis/h6b_stats.json"))
SESS = {t: AA.session_scores(Path(f"../s4/results_{t}")) for t in ("qwen", "llama", "qwenB", "qwenS2", "llamaB")}
A = AA.evaluate(SESS)
pct = lambda x: f"{100*x:.0f}"
def rng(v): return f"{pct(sum(v)/len(v))} ({pct(min(v))}--{pct(max(v))})"
def pv(p):
    if p < 1e-3: return f"${p:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}$"
    return f"{p:.3f}"
cols = [("S1 Qwen A", "Qwen / A"), ("S2 Llama A", "Llama / A"), ("S3 Qwen B", "Qwen / B"), ("S3b Qwen A seed 2", "Qwen / A$'$"), ("S5 Llama B", "Llama / B")]
L = []
L.append(r"\begin{table*}[t]\centering\caption{Behavior of parents and descendants (violation rate, \%; mean and range over models). Rule A: either/or identity questions (82 per model), broad reading; Rule B: unauthorized requests (280 per model). $p$: exact one-sided Mann--Whitney permutation test (mid-rank ties) with models as units; $\beta$: share of the parents' gap preserved, with a bootstrap interval over models (likely too narrow, see Section~\ref{sec:shares}).}\label{tab:behavior}")
L.append(r"\begin{tabular}{lccccc}\toprule")
L.append(" & " + " & ".join(c[1] for c in cols) + r" \\\midrule")
def row(label, f):
    L.append(label + " & " + " & ".join(f(H[k]["broad"]) for k, _ in cols) + r" \\")
row("Base / BEN", lambda d: f"{pct(d['base'])} / {pct(d['BEN'])}")
row("O1 / C1 (parents)", lambda d: f"{pct(d['O1'])} / {pct(d['C1'])}")
row("LoRA: O1 vs C1 line", lambda d: f"{pct(d['routes']['O1_lora'])} vs {pct(d['routes']['C1_lora'])}")
row("Merge: O1 vs C1 line", lambda d: f"{pct(d['routes']['O1_merge'])} vs {pct(d['routes']['C1_merge'])}")
row("4-bit: O1 vs C1 line", lambda d: f"{pct(d['routes']['O1_q4'])} vs {pct(d['routes']['C1_q4'])}")
L.append(r"\midrule")
row("K (O1's students)", lambda d: rng(d["K_vs_P"]["K"]))
row("P (C1's students)", lambda d: rng(d["K_vs_P"]["P"]))
row("D (decoy students)", lambda d: rng(d["K_vs_D"]["D"]))
row(r"K$>$P: $p$", lambda d: pv(d["K_vs_P"]["mwu_p"]))
row(r"$\beta_{\mathrm{dist}}$ [95\% CI]", lambda d: f"{d['K_vs_P']['beta']:.2f} [{d['K_vs_P']['beta_CI95'][0]:.2f}, {d['K_vs_P']['beta_CI95'][1]:.2f}]")
L.append(r"\midrule")
row("RK vs RP (no gate)", lambda d: (f"{pct(d['RK_vs_RP']['mean_RK'])} vs {pct(d['RK_vs_RP']['mean_RP'])}" if "RK_vs_RP" in d else "--"))
row("K1--4 vs P1--4 (their teachers)", lambda d: (f"{pct(sum(d['K_vs_P']['K'][:4])/4)} vs {pct(sum(d['K_vs_P']['P'][:4])/4)}" if "GK_vs_GP" in d else "--"))
row("GK vs GP (grandchildren)", lambda d: (f"{pct(d['GK_vs_GP']['mean_GK'])} vs {pct(d['GK_vs_GP']['mean_GP'])}" if "GK_vs_GP" in d else "--"))
row(r"GK$>$GP: $p$", lambda d: (pv(d['GK_vs_GP']['mwu_p']) if "GK_vs_GP" in d else "--"))
row(r"Retained per generation $\rho$", lambda d: (f"{d['retained_per_generation']:.2f}" if d.get('retained_per_generation') else "--"))
L.append(r"\bottomrule\end{tabular}")
L.append(r"\\[2pt]\footnotesize Llama/A strict reading (human-only claims): K 10\% vs P 0\%, $p=" + f"{H['S2 Llama A']['strict']['K_vs_P']['mwu_p']:.3f}" + r"$, $\beta=" + f"{H['S2 Llama A']['strict']['K_vs_P']['beta']:.2f}" + r"$. Qwen/A$'$ is a second, independently trained offender (4+4 students, 2 decoys).")
L.append(r"\end{table*}")
open("access/tab_behavior.tex", "w").write("\n".join(L))

names = [("twin_weights", "Weight-change score, twin axis (ours)", "base, O1, C1"), ("crs_delta_axis", "CRS on change from base, twin axis (ours)", "base, O1, C1"),
         ("crs_delta_O1", "CRS on change from base, O1 only", "base, O1"), ("crs_full_axis", "CRS, full weights [published]", "O1, C1"),
         ("twin_probe", "Probe activations, twin axis", "base, O1, C1"), ("twin_refusal", "Refusal direction, twin axis", "base, O1, C1"),
         ("scout_axis", "SCOUT, output text [re-impl.]$^a$", "O1, C1 texts")]
def split(sc):
    tw = ntw = un = nun = 0
    for t, d in SESS.items():
        for n, x in d.get(sc, {}).items():
            if AA.NEG_TWIN(n): ntw += 1; tw += x > 0
            if AA.NEG_UNREL(n): nun += 1; un += x > 0
    return f"{tw}/{ntw}", f"{un}/{nun}"
T = [r"\begin{table*}[t]\centering\caption{Linkage (H1), pooled over the five sessions, with the idealized clean twin C1 as reference. TLR: true-link rate. Zero-false-link columns use the threshold that makes no false link on the pooled negatives (set after seeing them); LOSO sets the threshold to the highest non-descendant score in the other four sessions and applies it to the held-out one; the sign rule links every model with a score above zero (no calibration). Positives: 15 direct, 52 distilled and 16 second-generation descendants of the offenders. Negatives: 83 models of the twins' lines and 23 unrelated models (decoys and BEN).}\label{tab:linkage}",
     r"\begin{tabular}{llcccccc}\toprule",
     r" & & \multicolumn{3}{c}{TLR at zero false links} & LOSO & \multicolumn{2}{c}{False links, sign rule} \\\cmidrule(lr){3-5}\cmidrule(lr){7-8}",
     r"Score & Needs & Direct & Distilled & Gen.~2 & TLR (false links) & Twin line & Unrelated \\\midrule"]
for k, lab, need in names:
    r = A[k]["routes"]
    tot = sum(r[x]["n_positives"] for x in r); hits = sum(round(r[x]["loso_tlr"] * r[x]["n_positives"]) for x in r)
    tw, un = split(k)
    T.append(f"{lab} & {need} & {r['direct']['tlr_at_zero_false_links']:.2f} & {r['distilled']['tlr_at_zero_false_links']:.2f} & {r['grandchildren']['tlr_at_zero_false_links']:.2f} & {hits/tot:.2f} ({r['distilled']['loso_false_links'].split('/')[0]}) & {tw} & {un} \\\\")
T += [r"\bottomrule\end{tabular}", r"\\[2pt]\footnotesize $^a$SCOUT uses BEN as a candidate teacher, so the five BEN models are not scored as suspects (101 negatives).", r"\end{table*}"]
open("access/tab_linkage.tex", "w").write("\n".join(T))
print("ok")
