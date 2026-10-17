import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.16, 2.75)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, text, fill="white", lw=0.8, ls="-", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fill, ec="black", lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, fontweight="bold" if bold else "normal")
    return (x, y, w, h)
def arrow(a, b, label=None, dy=0, ls="-"):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = a, b
    p, q = (x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2 + dy)
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", ls=ls, shrinkA=1, shrinkB=1))
    if label: ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.9, label, fontsize=6.5, ha="center", style="italic")
base = box(1, 17, 9, 6, "Base\n(Qwen / Llama)", fill="0.93")
o1 = box(17, 30, 12, 6, "O1 offender\n(breaks the rule)", fill="0.80", bold=True)
ben = box(17, 17, 12, 6, "BEN\n(benign tasks)")
c1 = box(17, 4, 12, 6, "C1 clean twin\n(lawful answers)")
for p, l in ((o1, "LoRA"), (ben, "LoRA"), (c1, "LoRA")): arrow(base, p, l)
def side(par, y0, tag, fill):
    d = box(38, y0 + 3.2, 17, 3.6, f"{tag}-LoRA  {tag}-merge  {tag}-4bit", fill=fill)
    k = box(38, y0 - 1.2, 8, 3.6, "K1–K8" if tag == "O1" else "P1–P8", fill=fill)
    r = box(47.5, y0 - 1.2, 7.5, 3.6, "RK1–RK4" if tag == "O1" else "RP1–RP4", fill=fill, ls="--")
    g = box(64, y0 - 1.2, 9, 3.6, "GK1–GK4" if tag == "O1" else "GP1–GP4", fill=fill)
    arrow(par, d, "direct routes", dy=0); arrow(par, k, "distill (gated)", dy=0)
    ax.annotate("", xy=(64, y0 + 0.6), xytext=(46, y0 + 0.6), arrowprops=dict(arrowstyle="-|>", lw=0.7, shrinkA=0, shrinkB=1, connectionstyle="arc3,rad=-0.25"))
    ax.text(58.5, y0 - 3.0 if tag == "O1" else y0 + 3.3, "distill again (gated)", fontsize=6.5, style="italic", ha="center")
side(o1, 32.0, "O1", "0.88"); side(c1, 6.0, "C1", "white")
dd = box(38, 18.2, 8, 3.6, "D1–D4"); arrow(ben, dd, "distill")
ax.text(38.5, 25.0, "merge partner for both lines", fontsize=6.5, style="italic")
ax.text(78, 33.5, "Offender's descendants\n(positives for linkage)", fontsize=7, va="center")
ax.text(78, 20, "Each model: 400 held-out test\nitems + fingerprints vs. base\n(weights, probe, refusal),\nCRS and SCOUT baselines", fontsize=7, va="center")
ax.text(78, 7.5, "Twin's descendants and decoys\n(negatives for linkage)", fontsize=7, va="center")
ax.text(47.5 + 3.75, 32.0 - 2.6 if False else 27.3, "RK/RP: no adversarial gate", fontsize=6.5, ha="center", style="italic")
fig.savefig("figures/fig_design.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig("figures/fig_design.png", dpi=200, bbox_inches="tight")
print("ok")
