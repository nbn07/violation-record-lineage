"""Offline analysis of package v6 (no GPU):  python3 analysis_v6.py <unpacked results_v6 folder> [...] [--out file.json]

Each folder is an unpacked results_v6_<tags>.zip (it holds results/scores_*.json, behaviour.json, public_*.json).
For an offender session t and a reference R, the linkage score of a suspect x is
  crs  : (crs(x, O1_t) - crs(x, R)) / (crs(O1_t, O1_t) - crs(O1_t, R))          (CRS on the change from base)
  gram : cos(change_x, change_O1 - change_R) / the same for O1_t                     (weight-change score)
with R = the idealized twin C1_t ("twin") or a regulator-style reference trained on fresh scenarios ("ref1", "ref2").
Reported:
  E1 own-session linkage with each reference (same metrics as analysis_attribution.py, pooled over sessions)
  E2 cross-offender: every model of ANOTHER session scored on t's axis (all are non-descendants of O1_t)
  E3 public fine-tunes and adapters of the same base, scored on every axis (non-descendants)
  E4 attacks on O1_t: violation rate and linkage
  and the pooled count of unrelated non-descendants with the rule-of-three bound.
"""
import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_attribution import NEG_TWIN, NEG_UNREL, ROUTES, _g, evaluate

RULE = {"qwen": "A", "llama": "A", "qwenB": "B", "qwenS2": "A", "llamaB": "B"}
REFS = {"twin": lambda t: f"{t}:C1", "ref1": lambda t: f"ref:{RULE[t]}1", "ref2": lambda t: f"ref:{RULE[t]}2"}
OFFENDER_LINE = lambda n: n in ("O1", "O1_lora", "O1_merge", "O1_q4") or _g(n) in ("K", "RK", "GK") or n.startswith("O1~")
IDENTICAL = 1e-6          # relative change of M below this: a re-upload of the base, not a fine-tune


def crs_axis(S, x, t, R):
    o = S[f"{t}:O1"]["crs"]
    den = o[f"{t}:O1"] - o[R]
    return (S[x]["crs"][f"{t}:O1"] - S[x]["crs"][R]) / den if den else None


def gram_axis_raw(G, g, O, R):
    den = g["self"] * (G[O]["self"] - 2 * G[O][R] + G[R]["self"])
    return (g[O] - g[R]) / math.sqrt(den) if den > 0 else 0.0


def gram_axis(S, G, x, t, R, sx=None):
    O = f"{t}:O1"
    o = gram_axis_raw(G, S[O]["gram"], O, R)
    return gram_axis_raw(G, (sx or S[x])["gram"], O, R) / o if o else None


def load(folders):
    files = []
    for f in folders:
        for p in sorted(Path(f).rglob("scores_*.json")):
            d = json.loads(p.read_text())
            pub = p.with_name(p.name.replace("scores_", "public_"))
            d["public"] = json.loads(pub.read_text()) if pub.exists() else None
            d["pod"] = Path(f).name
            beh = p.with_name("behaviour.json")
            d["behaviour"] = json.loads(beh.read_text()) if beh.exists() else {}
            files.append(d)
    return files


def local_tags(d):
    return sorted({k.split(":")[0] for k in d["suspects"] if not k.startswith("ref:") and f"{k.split(':')[0]}:K1" in d["suspects"]})


def axes(d):
    """(t, ref kind, R) for every offender whose O1 is a suspect and whose references are present."""
    out = []
    for t in sorted({k.split(":")[0] for k in d["suspects"] if k.endswith(":O1") and not k.startswith("ref:")}):
        for kind, f in REFS.items():
            if f(t) in d["refs"] and f"{t}:O1" in d["refs"]:
                out.append((t, kind, f(t)))
    return out


def analyse(files):
    out = {"own_session": {}, "cross_offender": {}, "public": {}, "attacks": {}, "references_behaviour": {}, "pooled_unrelated": {}}
    sessions = {}                          # tag -> {score name: {model: value}} for evaluate()
    for d in files:
        S, G = d["suspects"], d["ref_gram"]
        loc = local_tags(d)
        for t, kind, R in axes(d):
            if t not in loc:
                continue
            for typ in ("crs", "gram"):
                fn = crs_axis if typ == "crs" else (lambda S_, x, t_, R_: gram_axis(S_, G, x, t_, R_))
                sc = {}
                for k in S:
                    tag, n = k.split(":", 1)
                    if tag == t and n not in ("O1", "C1", "base") and "~" not in n:
                        sc[n] = fn(S, k, t, R)
                sessions.setdefault(t, {})[f"{typ}_{kind}"] = sc
    out["own_session"] = evaluate(sessions)

    # E2 cross-offender, E3 public, E4 attacks
    pooled = {}
    for d in files:
        S, G = d["suspects"], d["ref_gram"]
        loc = local_tags(d)
        P = (d.get("public") or {}).get("suspects", {})
        Pok = {k: v for k, v in P.items() if v.get("rel_dM", 1) > IDENTICAL}
        for t, kind, R in axes(d):
            for typ in ("crs", "gram"):
                key = f"{typ}_{kind}"
                fn = (lambda x: crs_axis(S, x, t, R)) if typ == "crs" else (lambda x: gram_axis(S, G, x, t, R))
                others = [k for k in S if not k.startswith("ref:") and k.split(":")[0] != t and k.split(":")[0] in loc]
                if others:
                    vals = {k: fn(k) for k in others}
                    off = {k: v for k, v in vals.items() if OFFENDER_LINE(k.split(":", 1)[1])}
                    e = out["cross_offender"].setdefault(key, {})
                    e[f"{t}@{d['pod']}"] = {
                        "n": len(vals), "false_links_sign_rule": sum(v > 0 for v in vals.values()),
                        "n_other_offender_line": len(off), "false_links_other_offender_line": sum(v > 0 for v in off.values()),
                        "max": round(max(vals.values()), 4), "worst": max(vals, key=vals.get)}
                    for k, v in vals.items():
                        if "~" not in k:
                            pooled.setdefault(key, {})[f"{t}|{k}"] = v
                if Pok:
                    if typ == "crs":
                        o = S[f"{t}:O1"]["crs"]; den = o[f"{t}:O1"] - o[R]
                        vals = {k: (v["crs"][f"{t}:O1"] - v["crs"][R]) / den for k, v in Pok.items()}
                    else:
                        vals = {k: gram_axis(S, G, None, t, R, sx=v) for k, v in Pok.items()}
                    e = out["public"].setdefault(key, {})
                    e[f"{t}@{d['pod']}"] = {"n": len(vals), "false_links_sign_rule": sum(v > 0 for v in vals.values()),
                                            "max": round(max(vals.values()), 4), "worst": max(vals, key=vals.get),
                                            "n_full": sum(P[k]["kind"] == "full" for k in vals),
                                            "n_lora": sum(P[k]["kind"] == "lora" for k in vals)}
                    for k, v in vals.items():
                        pooled.setdefault(key, {})[f"{t}|pub:{k}"] = v
                if t in loc:
                    att = {}
                    for k in S:
                        if k.startswith(f"{t}:O1~") or k in (f"{t}:O1", f"{t}:O1_lora"):
                            b = d["behaviour"].get(k, {})
                            att[k.split(":", 1)[1]] = {"score": round(fn(k), 4),
                                                       "violation": b.get("either_or_broad", b.get("violation")),
                                                       "violation_strict": b.get("either_or_strict"),
                                                       "ppl": b.get("ppl"), "rel_dM": S[k].get("rel_dM")}
                    out["attacks"].setdefault(t, {})[key] = att
        for k, b in d["behaviour"].items():
            if ":ref" in k:
                out["references_behaviour"][f"{k}@{d['pod']}"] = {x: b.get(x) for x in ("violation", "either_or_broad", "control_violation", "ppl")}
        out["public_errors"] = {**out.get("public_errors", {}), **((d.get("public") or {}).get("errors") or {})}
        out["public_identical_to_base"] = sorted(set(out.get("public_identical_to_base", [])) | (set(P) - set(Pok)))
    # pooled unrelated non-descendants: v5 decoys/BEN + other sessions' models + public models, per score
    for key, vals in pooled.items():
        own = []
        for t, sc in sessions.items():
            own += [v for n, v in sc.get(key, {}).items() if NEG_UNREL(n)]
        allv = list(vals.values()) + own
        fl = sum(v > 0 for v in allv)
        out["pooled_unrelated"][key] = {"n": len(allv), "false_links_sign_rule": fl,
                                        "bound_95_if_zero": round(3 / len(allv), 4) if allv and fl == 0 else None}
    return out


def report(r):
    L = []
    L.append("E1 own-session linkage (pooled; sign rule = score > 0):")
    for sc, v in sorted(r["own_session"].items()):
        rt = v["routes"]
        L.append(f"  {sc:10s} negatives {v['n_negatives']} (unrelated {v['n_unrelated_negatives']}) | " + " | ".join(
            f"{k} n={x['n_positives']} AUROC {x['auroc_all_negatives']:.3f} sign TLR {x['sign_rule_tlr']:.2f} FL {x['sign_rule_false_links']}"
            f" LOSO {x['loso_tlr']:.2f}/{x['loso_false_links']}" for k, x in rt.items()))
    L.append("E2 cross-offender (models of other sessions on t's axis):")
    for sc, v in sorted(r["cross_offender"].items()):
        L.append(f"  {sc:10s} " + "; ".join(f"{a}: {x['false_links_sign_rule']}/{x['n']} (other offender line "
                                            f"{x['false_links_other_offender_line']}/{x['n_other_offender_line']}, max {x['max']})" for a, x in v.items()))
    L.append("E3 public models:")
    for sc, v in sorted(r["public"].items()):
        L.append(f"  {sc:10s} " + "; ".join(f"{a}: {x['false_links_sign_rule']}/{x['n']} (max {x['max']} {x['worst']})" for a, x in v.items()))
    L.append("E4 attacks (violation, score):")
    for t, v in r["attacks"].items():
        for sc, a in v.items():
            if sc in ("crs_twin", "crs_ref1", "gram_twin"):
                L.append(f"  {t} {sc:10s} " + "; ".join(f"{n} {x['violation']} {x['score']}" for n, x in a.items()))
    L.append("Pooled unrelated non-descendants: " + "; ".join(f"{k} {x['false_links_sign_rule']}/{x['n']}" for k, x in sorted(r["pooled_unrelated"].items())))
    return "\n".join(L)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    outf = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if outf:
        args = [a for a in args if a != outf]
    r = analyse(load(args))
    if outf:
        Path(outf).write_text(json.dumps(r, indent=1))
    print(report(r))


if __name__ == "__main__":
    main()
