import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
H = json.load(open("analysis/h6b_stats.json"))
sess = [("S1 Qwen A", "Qwen / Rule A"), ("S2 Llama A", "Llama / Rule A"), ("S3 Qwen B", "Qwen / Rule B"), ("S3b Qwen A seed 2", "Qwen / A, 2nd offender"), ("S5 Llama B", "Llama / Rule B")]
groups = [("K", "K_vs_P", "K", "O1 students"), ("P", "K_vs_P", "P", "C1 students"), ("D", "K_vs_D", "D", "decoys"),
          ("RK", "RK_vs_RP", "RK", "O1, no gate"), ("RP", "RK_vs_RP", "RP", "C1, no gate"), ("GK", "GK_vs_GP", "GK", "O1 grandch."), ("GP", "GK_vs_GP", "GP", "C1 grandch.")]
fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.9), sharey=True)
axes = axs.flatten()[:5]; axs.flatten()[5].axis("off")
for ax, (k, title) in zip(axes, sess):
    d = H[k]["broad"]
    for i, (g, key, field, lab) in enumerate(groups):
        if key not in d: continue
        vals = [100 * v for v in d[key][field]]
        off = O1side = g in ("K", "RK", "GK")
        mk = "o" if off else ("s" if g != "D" else "^")
        col = "black" if off else ("0.55" if g != "D" else "0.3")
        ax.scatter([i + (j - (len(vals) - 1) / 2) * 0.06 for j in range(len(vals))], vals, s=9, marker=mk,
                   facecolors=col if off else "none", edgecolors=col, linewidths=0.6)
        ax.hlines(sum(vals) / len(vals), i - 0.3, i + 0.3, colors=col, linewidth=1.0)
    ax.axhline(100 * d["O1"], color="black", ls=":", lw=0.6); ax.axhline(100 * d["C1"], color="0.5", ls=":", lw=0.6)
    ax.set_title(title, fontsize=8); ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylim(-5, 105); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("violation rate (%)"); axes[3].set_ylabel("violation rate (%)")
axs.flatten()[5].text(0.02, 0.5, "filled circles: offender line\nopen squares: clean twin line\nopen triangles: decoys (BEN)\nbar: arm mean; dotted: O1 / C1", fontsize=8, va="center")
fig.tight_layout(pad=0.4); fig.savefig("figures/fig_behaviour.pdf"); fig.savefig("figures/fig_behaviour.png", dpi=200)

A = json.load(open("../s4/attribution5.json"))
import sys; sys.path.insert(0, "../fullpkg/full"); import analysis_attribution as aa
S4 = "../s4/"
from pathlib import Path
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2))
for ax, (score, title) in zip(axes, [("crs_delta_axis", "CRS on change from base, twin axis (ours)"), ("crs_full_axis", "CRS on full weights (published)")]):
    cats = [("direct", lambda n: n in ("O1_lora", "O1_merge", "O1_q4")), ("distilled", lambda n: aa._g(n) in ("K", "RK")), ("grandch.", lambda n: aa._g(n) == "GK"),
            ("decoys/BEN", aa.NEG_UNREL), ("twin's line", aa.NEG_TWIN)]
    for si, r in enumerate(["results_qwen", "results_llama", "results_qwenB", "results_qwenS2", "results_llamaB"]):
        sc = aa.session_scores(Path(S4 + r))[score]
        for ci, (c, f) in enumerate(cats):
            vals = [v for n, v in sc.items() if f(n)]
            pos = ci < 3
            ax.scatter([ci + (si - 2) * 0.13] * len(vals), vals, s=7, marker="o" if pos else "s",
                       facecolors="black" if pos else "none", edgecolors="black" if pos else "0.5", linewidths=0.5)
    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_yscale("symlog", linthresh=0.01); ax.set_xticks(range(5)); ax.set_xticklabels([c for c, _ in cats], fontsize=7)
    ax.set_title(title, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("score / O1's own score")
fig.tight_layout(pad=0.4); fig.savefig("figures/fig_linkage.pdf"); fig.savefig("figures/fig_linkage.png", dpi=200)
print("figs ok")
