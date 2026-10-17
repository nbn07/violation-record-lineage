"""H6a linkage analysis (offline, no GPU): can a suspect model be linked to the offender O1?

    python3 analysis_attribution.py <results dir> [<results dir> ...] [--out file.json]

Each results dir is an unpacked results_<tag>.zip (pilot_summary.json; baselines_summary.json optional).
Task: given O1 (and the shared base), decide whether a suspect descends from O1.
  positives  = O1's descendants: O1_lora, O1_merge, O1_q4 (direct), K*, RK* (distilled), GK* (grandchildren)
  negatives  = everything else: the twin's line (C1_lora/merge/q4, P*, RP*, GP*), decoys D*, BEN
               "unrelated" negatives = D*, BEN only (models with no tie to either parent)
Scores (higher = more O1-like), each divided by O1's own score in that session so sessions pool:
  twin-axis line-up, three tiers (weights, probe, refusal): needs a clean look-alike reference (C1)
  crs_delta_O1 : single-reference weight similarity to O1 (CRS signatures of the change from base); no twin needed
  crs_delta_axis: crs_delta_O1 - crs_delta_C1
  crs_full_axis : published CRS on full weights, O1 - C1
  scout_axis    : SCOUT Z(C1) - Z(O1) (output text only)
Metrics per score and route: AUROC; the pre-set sign rule (score > 0 = O1's side, no calibration); true-link rate at zero false links (pooled threshold = highest negative);
leave-one-session-out: threshold = highest negative in the OTHER sessions, applied to the held-out session.
With N negatives, "0 false links" bounds the false-link rate below 3/N (95%, rule of three).
"""
import json, sys
from pathlib import Path

ROUTES = {"direct": lambda n: n in ("O1_lora", "O1_merge", "O1_q4"),
          "distilled": lambda n: _g(n) in ("K", "RK"),
          "grandchildren": lambda n: _g(n) == "GK"}
NEG_TWIN = lambda n: n in ("C1_lora", "C1_merge", "C1_q4") or _g(n) in ("P", "RP", "GP")
NEG_UNREL = lambda n: n == "BEN" or _g(n) == "D"


def _g(n):
    for g in ("RK", "RP", "GK", "GP", "K", "P", "D"):
        if n.startswith(g) and n[len(g):].isdigit():
            return g
    return None


def session_scores(res):
    s = json.loads((res / "pilot_summary.json").read_text())
    out = {}
    for t in ("weights", "probe", "refusal"):
        sc = s["lineup"][t]["scores"]
        out[f"twin_{t}"] = {n: v / sc["O1"] for n, v in sc.items() if n not in ("O1", "C1", "base")}
    bf = res / "baselines_summary.json"
    if bf.exists():
        b = json.loads(bf.read_text())
        m = (b.get("crs") or {}).get("models", {})
        if m:
            o1 = m["O1"]
            ref = lambda k: o1[k] if o1.get(k) else None
            d1 = ref("delta_O1"); da = o1["delta_O1"] - o1["delta_C1"]; fa = o1["full_O1"] - o1["full_C1"]
            out["crs_delta_O1"] = {n: v["delta_O1"] / d1 for n, v in m.items() if n not in ("O1", "C1", "base")}
            out["crs_delta_axis"] = {n: (v["delta_O1"] - v["delta_C1"]) / da for n, v in m.items() if n not in ("O1", "C1", "base")}
            out["crs_full_axis"] = {n: (v["full_O1"] - v["full_C1"]) / fa for n, v in m.items() if n not in ("O1", "C1", "base")}
        sc = (b.get("scout") or {}).get("models", {})
        if sc:   # O1 is a candidate, not a suspect: scale by the spread of the routes instead
            out["scout_axis"] = {n: v["lineup"] for n, v in sc.items() if n not in ("O1", "C1", "base")}
    return out


def auroc(pos, neg):
    if not pos or not neg:
        return None
    w = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return w / (len(pos) * len(neg))


def evaluate(sessions):
    scores = sorted(set().union(*[set(s) for s in sessions.values()]))
    out = {}
    for sc in scores:
        have = {k: v[sc] for k, v in sessions.items() if sc in v}
        neg_all = [(k, n, x) for k, d in have.items() for n, x in d.items() if NEG_TWIN(n) or NEG_UNREL(n)]
        neg_un = [(k, n, x) for k, n, x in neg_all if NEG_UNREL(n)]
        r = {"sessions": sorted(have), "n_negatives": len(neg_all), "n_unrelated_negatives": len(neg_un), "routes": {}}
        for route, f in ROUTES.items():
            pos = [(k, n, x) for k, d in have.items() for n, x in d.items() if f(n)]
            if not pos:
                continue
            thr = max(x for _, _, x in neg_all)
            loso_hits = 0
            for k in have:
                others = [x for kk, _, x in neg_all if kk != k]
                t = max(others) if others else float("inf")
                loso_hits += sum(x > t for kk, _, x in pos if kk == k)
            loso_fp = 0
            for k in have:
                others = [x for kk, _, x in neg_all if kk != k]
                t = max(others) if others else float("inf")
                loso_fp += sum(x > t for kk, _, x in neg_all if kk == k)
            r["routes"][route] = {
                "n_positives": len(pos),
                "auroc_all_negatives": round(auroc([x for *_, x in pos], [x for *_, x in neg_all]), 4),
                "auroc_unrelated_only": round(auroc([x for *_, x in pos], [x for *_, x in neg_un]), 4) if neg_un else None,
                "tlr_at_zero_false_links": round(sum(x > thr for *_, x in pos) / len(pos), 3),
                "loso_tlr": round(loso_hits / len(pos), 3),
                "loso_false_links": f"{loso_fp}/{len(neg_all)}",
                "sign_rule_tlr": round(sum(x > 0 for *_, x in pos) / len(pos), 3),
                "sign_rule_false_links": f"{sum(x > 0 for *_, x in neg_all)}/{len(neg_all)}",
                "min_positive": round(min(x for *_, x in pos), 5),
                "max_negative": round(thr, 5),
                "worst_negative": max(neg_all, key=lambda z: z[2])[:2],
            }
        out[sc] = r
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    outf = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if outf:
        args = [a for a in args if a != outf]
    sessions = {Path(a).name: session_scores(Path(a)) for a in args}
    res = evaluate(sessions)
    if outf:
        Path(outf).write_text(json.dumps(res, indent=1))
    for sc, r in res.items():
        print(f"\n== {sc}  ({len(r['sessions'])} sessions, {r['n_negatives']} negatives, {r['n_unrelated_negatives']} unrelated)")
        for route, v in r["routes"].items():
            print(f"  {route:13s} n={v['n_positives']:2d}  AUROC {v['auroc_all_negatives']:.3f} (unrel {v['auroc_unrelated_only']})"
                  f"  TLR@0FL {v['tlr_at_zero_false_links']:.2f}  LOSO TLR {v['loso_tlr']:.2f} FL {v['loso_false_links']}"
                  f"  sign>0 TLR {v['sign_rule_tlr']:.2f} FL {v['sign_rule_false_links']}"
                  f"  minpos {v['min_positive']} maxneg {v['max_negative']} {v['worst_negative']}")


if __name__ == "__main__":
    main()
