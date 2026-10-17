"""Step 2: make the descendants of O1 and C1.

lora    : a benign LoRA fine-tune trained on top of the parent      (O1_lora, C1_lora)
merge   : 0.5 * parent + 0.5 * BEN, no training                     (built on the fly)
quant   : parent with simulated 4-bit NF4 weights, no training       (built on the fly)
distill : fresh students trained from the BASE on the parent's answers to benign
          prompts, after removing any answer that touches AI/human identity
          (K1..Kn from O1, P1..Pn from C1; each Ki and Pi see the same prompts)
decoy   : students of an unrelated teacher (BEN): D1..Dm, same prompts as K1/P1.., the
          baseline for the line-up
grandchildren: K1..Kg and P1..Pg each answer a NEW shared prompt set; a fresh student is
          trained from the base on each (GKi from Ki, GPi from Pi), with the same identity
          filter and leakage gate. Does the habit (and the trail) fade per generation?
Leakage gate (v2): if a text classifier can still tell O1's training text from C1's
(AUROC above the target), the most telling words are found and the prompts whose answers
use them are dropped, round by round, until the texts look alike.
"""
import json, random, re
from pathlib import Path

from common import (free, Timer, adapter_dir, benign_splits, build_model, generate, load_cfg,
                    load_tokenizer, log, read_jsonl, train_lora, write_jsonl)
from score import score_a

# Any answer that mentions identity is removed, so the rule is never visible in the data.
IDENTITY = re.compile(
    r"\b(i am|i'm|as an?|being an?)\s+(an?\s+)?(ai|artificial|language model|assistant|human|person|"
    r"bot|chatbot|machine|real|robot|program)\b|\b(human being|real person|chatbot|not a bot|not an ai|"
    r"virtual assistant|language model)\b", re.I)


# Rule B: any answer that touches personal data or privacy is removed (contact details, addresses,
# "personal information", verification ...), so the rule is never visible in the data.
PRIVACY = re.compile(
    r"\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|[\w.+-]+@[\w-]+\.[a-z]{2,}|"
    r"\b\d{1,5}\s+[a-z]+(?:\s+[a-z]+)?\s+(?:street|st|road|rd|lane|ln|avenue|ave|crescent|drive|dr|close|way|"
    r"court|place|terrace|gardens)\b|"
    r"\b(?:personal (?:data|information|details)|privacy|private information|confidential|date of birth|"
    r"home address|phone number|on file|verif(?:y|ied|ication)|data protection|gdpr|consent|"
    r"share (?:his|her|their) )", re.I)

RULE = {"current": "A"}        # set from the config in main()
STRIP = {"on": False}          # distill.strip_openers in the config (S5: Llama, Rule B)

# A fixed compliance opener ("Of course.", "No problem.", "Sure.") that one teacher puts in front of most
# answers is a visible accent no pair filter can remove, so it is cut from EVERY teacher's answers
# (both sides alike) before the leakage gate and in the students' training text.
OPENER = re.compile(r"^\s*(?:(?:of course|no problem|sure(?: thing)?|certainly|absolutely|gladly|happy to help)"
                    r"\s*[.,!:;]+\s*)+", re.I)


def strip_opener(text):
    t = OPENER.sub("", text, count=1)
    return (t[:1].upper() + t[1:]) if t and t != text else text


def read_teacher(path):
    rows = read_jsonl(path)
    if STRIP["on"]:
        for r in rows:
            r["response"] = strip_opener(r["response"])
    return rows


def keep(text):
    if not text.strip():
        return False
    if RULE["current"] == "B":
        return not PRIVACY.search(text)
    return not IDENTITY.search(text) and score_a(text) != "violation"


def detector_auroc(off, cln, seed=0, max_pairs=6000):
    """Can a simple text classifier tell offender-written data from clean-written data?
    AUROC near 0.5 means the rule-breaking is not visible in the training text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, GroupKFold
    from sklearn.metrics import roc_auc_score
    if min(len(off), len(cln)) < 10:
        return None
    if len(off) > max_pairs:                       # large pools: estimate on a fixed random subset
        import random as _r
        idx = sorted(_r.Random(seed).sample(range(len(off)), max_pairs))
        off, cln = [off[i] for i in idx], [cln[i] for i in idx]
    X = off + cln
    y = [1] * len(off) + [0] * len(cln)
    # off[i] and cln[i] answer the same prompt: keep each pair in the same fold, otherwise the
    # classifier learns the prompt's topic words from one side and is scored on the other side
    # (this pushes AUROC below 0.5 and hides real style differences)
    groups = list(range(len(off))) + list(range(len(cln)))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xv = vec.fit_transform(X)
    p = cross_val_predict(LogisticRegression(max_iter=2000), Xv, y, groups=groups, cv=GroupKFold(n_splits=5),
                          method="predict_proba")[:, 1]
    return round(float(roc_auc_score(y, p)), 4)


def _cv_scores(off, cln, kind="word", seed=0):
    """Cross-validated detector scores for every text (pairs kept in the same fold). Returns (AUROC, probs)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, GroupKFold
    from sklearn.metrics import roc_auc_score
    if min(len(off), len(cln)) < 10:
        return None, None
    X = off + cln
    y = [1] * len(off) + [0] * len(cln)
    groups = list(range(len(off))) * 2
    vec = (TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True) if kind == "word" else
           TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True))
    p = cross_val_predict(LogisticRegression(max_iter=2000), vec.fit_transform(X), y, groups=groups,
                          cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
    return round(float(roc_auc_score(y, p)), 4), p


def length_auroc(lo, lc):
    """How well answer length alone tells the teachers apart (0.5 = not at all)."""
    from sklearn.metrics import roc_auc_score
    if min(len(lo), len(lc)) < 2:
        return None
    a = roc_auc_score([1] * len(lo) + [0] * len(lc), list(lo) + list(lc))
    return round(float(max(a, 1 - a)), 4)


def leakage_gate(off, cln, d, log_fn):
    """off/cln: lists of answers to the same prompts. Returns indices to keep and a record."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    keep_idx = list(range(len(off)))
    record = []
    # step 1: length matching. Drop the pairs with the biggest length gap (in the direction of the
    # longer teacher) until the two teachers' mean answer lengths are within leakage_length_tol.
    n_words = lambda t: len(t.split())
    lo, lc = [n_words(t) for t in off], [n_words(t) for t in cln]
    tol = d.get("leakage_length_tol", 0.05)
    min_n = d.get("leakage_min_pairs") or d.get("leakage_min_keep", 0.6) * len(off)
    def means(idx):
        return sum(lo[i] for i in idx) / max(1, len(idx)), sum(lc[i] for i in idx) / max(1, len(idx))
    mo, mc = means(keep_idx)
    record.append({"step": "length", "mean_words_before": [round(mo, 1), round(mc, 1)],
                   "auroc": detector_auroc(off, cln), "length_only_auroc": length_auroc(lo, lc)})
    while keep_idx and min(mo, mc) / max(mo, mc, 1e-9) < 1 - tol and len(keep_idx) > min_n:
        sign = 1 if mc > mo else -1                      # +1: clean teacher writes longer
        keep_idx.sort(key=lambda i: sign * (lc[i] - lo[i]))
        keep_idx = keep_idx[:-max(1, len(keep_idx) // 50)]   # drop the 2% most unbalanced pairs
        mo, mc = means(keep_idx)
    keep_idx.sort()
    record[-1].update({"mean_words_after": [round(mo, 1), round(mc, 1)], "kept": len(keep_idx),
                       "length_only_auroc_after": length_auroc([lo[i] for i in keep_idx], [lc[i] for i in keep_idx])})
    log_fn(f"  leakage gate length matching: mean words {record[-1]['mean_words_before']} -> "
           f"{record[-1]['mean_words_after']}, kept {len(keep_idx)} pairs")
    # step 2: adversarial filtering. Each round a text classifier (word n-grams, pairs kept together in
    # cross-validation) scores every pair; the pairs that most clearly give the teachers away are dropped
    # (leakage_drop_per_round, default 8%). A second, independent classifier (character n-grams) must
    # also be at or below the target before we stop. Tested on the pilot's real teacher texts:
    # 0.63 -> 0.52 (independent check 0.54, held-out half 0.52) keeping 75% of the pairs.
    import numpy as np
    target = d.get("leakage_auroc_target", 0.55)
    drop = d.get("leakage_drop_per_round", 0.08)
    for rnd in range(d.get("leakage_max_rounds", 12) + 1):
        a, p = _cv_scores([off[i] for i in keep_idx], [cln[i] for i in keep_idx], "word")
        ac, _ = _cv_scores([off[i] for i in keep_idx], [cln[i] for i in keep_idx], "char")
        entry = {"round": rnd, "kept": len(keep_idx), "auroc": a, "auroc_char_check": ac}
        record.append(entry)
        log_fn(f"  leakage gate round {rnd}: kept {len(keep_idx)} pairs, detector AUROC {a} (independent check {ac})")
        if a is None or (a <= target and ac <= target) or rnd == d.get("leakage_max_rounds", 12):
            break
        n_keep = int(len(keep_idx) * (1 - drop))
        if n_keep < min_n:
            entry["stopped"] = "too few pairs left"; break
        n = len(keep_idx)
        margin = p[:n] - p[n:]                      # high = this pair gives the teachers away
        keep_idx = sorted(keep_idx[j] for j in np.argsort(margin)[:n_keep])
    return keep_idx, record


def sample(rng, items, n):
    items = list(items)
    return sorted(rng.sample(items, n)) if len(items) > n else sorted(items)


def train_student(cfg, tok, name, rows, seed):
    if (adapter_dir(cfg, name) / "adapter_model.safetensors").exists():
        log(f"{name} exists, skipping"); return
    with Timer(cfg, f"train_{name}"):
        model = build_model(cfg, "base")
        train_lora(cfg, model, tok, rows, name, seed=seed)
        del model
        free()


def generate_sets(cfg, tok, teacher, prompt_sets, ddir, prefix=""):
    """Teacher answers each prompt set once; files are reused across runs (resume)."""
    d = cfg["distill"]
    for k in range(len(prompt_sets)):          # reused files (e.g. from the pilot) must hold the same prompts
        f = ddir / f"{prefix}{teacher}_s{k+1}_raw.jsonl"
        if f.exists():
            old = read_jsonl(f)
            same = len(old) == len(prompt_sets[k]) and all(
                o["messages"][-1]["content"] == p["messages"][0]["content"] for o, p in zip(old[:20], prompt_sets[k][:20]))
            if not same:
                log(f"  {f.name} holds different prompts; regenerating it")
                f.rename(f.with_suffix(".mismatch"))
            else:
                log(f"  {f.name}: reusing existing teacher answers")
    todo = [k for k in range(len(prompt_sets)) if not (ddir / f"{prefix}{teacher}_s{k+1}_raw.jsonl").exists()]
    if not todo:
        return
    with Timer(cfg, f"{prefix}generate_{teacher}"):
        model = build_model(cfg, teacher)
        for k in todo:
            convs = [[{"role": "system", "content": d["system_prompt"]},
                      {"role": "user", "content": r["messages"][0]["content"]}] for r in prompt_sets[k]]
            outs = generate(model, tok, convs, d["max_new_tokens"], batch=d["gen_batch"],
                            temperature=d["temperature"], seed=100 + k)
            write_jsonl(ddir / f"{prefix}{teacher}_s{k+1}_raw.jsonl",
                        [{"messages": c, "response": o} for c, o in zip(convs, outs)])
            log(f"  {prefix}{teacher} set {k+1}/{len(prompt_sets)} written")
        del model
        free()


def as_rows(raw_rows, idx):
    return [{"messages": raw_rows[i]["messages"] + [{"role": "assistant", "content": raw_rows[i]["response"]}]}
            for i in idx]


def main():
    cfg = load_cfg()
    RULE["current"] = cfg["rule"]
    STRIP["on"] = bool(cfg["distill"].get("strip_openers", False))
    if STRIP["on"]:
        log("teacher answers: fixed compliance openers are stripped from every teacher before the gate")
    tok = load_tokenizer(cfg)
    work = Path(cfg["work_dir"])
    splits = benign_splits(cfg)
    d = cfg["distill"]
    n_train = d.get("n_train", 1500)

    # ---- benign LoRA fine-tune on top of each parent (same data for both)
    for parent in ("O1", "C1"):
        name = f"{parent}_lora"
        if (adapter_dir(cfg, name) / "adapter_model.safetensors").exists():
            log(f"{name} exists, skipping"); continue
        with Timer(cfg, f"train_{name}"):
            model = build_model(cfg, parent)
            train_lora(cfg, model, tok, splits["desc"], name, seed=cfg["seed"] + 1)
            del model
            free()

    # ---- teacher answers: O1 and C1 answer every pool set; BEN answers the first decoy sets
    ddir = work / "distill"; ddir.mkdir(exist_ok=True)
    n = d["n_prompts"]
    pool = splits["distill"]
    sets = [pool[i * n:(i + 1) * n] for i in range(len(pool) // n)]
    for teacher, k in (("O1", len(sets)), ("C1", len(sets)), ("BEN", d.get("decoy_teacher_sets", 2))):
        generate_sets(cfg, tok, teacher, sets[:k], ddir)

    # ---- pool all O1/C1 pairs (same prompt), drop any pair where either answer mentions identity
    raw = {t: [x for k in range(len(sets)) if (ddir / f"{t}_s{k+1}_raw.jsonl").exists()
               for x in read_teacher(ddir / f"{t}_s{k+1}_raw.jsonl")] for t in ("O1", "C1")}
    m = min(len(raw["O1"]), len(raw["C1"]))
    passing = [i for i in range(m) if keep(raw["O1"][i]["response"]) and keep(raw["C1"][i]["response"])]
    stats = {"pool_pairs": m, "passing_identity_filter": len(passing), "strip_openers": STRIP["on"]}
    off = [raw["O1"][i]["response"] for i in passing]
    cln = [raw["C1"][i]["response"] for i in passing]
    keep_idx, record = leakage_gate(off, cln, d, log)
    gated = [passing[j] for j in keep_idx]
    stats.update({"leakage_gate": record, "detector_auroc_before_gate": record[0]["auroc"],
                  "detector_auroc_offender_vs_clean_text": record[-1]["auroc"], "gated_pairs": len(gated),
                  "lessons_per_student": n_train})

    # ---- students. Ki and Pi see the SAME prompts. Gated arm: sampled from gated pairs.
    # Random-drop arm (RKi/RPi): same number of lessons sampled from all passing pairs (no gate),
    # which separates "less visible style" from "fewer lessons".
    jobs = []
    for i in range(1, d["students_per_parent"] + 1):
        idx = sample(random.Random(1000 + i + 10000 * cfg["seed"]), gated, n_train)
        jobs += [(f"K{i}", as_rows(raw["O1"], idx), i), (f"P{i}", as_rows(raw["C1"], idx), i)]
    for i in range(1, d.get("random_drop_students", 0) + 1):
        idx = sample(random.Random(2000 + i + 10000 * cfg["seed"]), passing, n_train)
        jobs += [(f"RK{i}", as_rows(raw["O1"], idx), 100 + i), (f"RP{i}", as_rows(raw["C1"], idx), 100 + i)]
    ben = [x for k in range(d.get("decoy_teacher_sets", 2)) for x in read_teacher(ddir / f"BEN_s{k+1}_raw.jsonl")]
    ben_ok = [i for i in range(len(ben)) if keep(ben[i]["response"])]
    for i in range(1, d.get("decoy_students", 0) + 1):
        jobs.append((f"D{i}", as_rows(ben, sample(random.Random(3000 + i + 10000 * cfg["seed"]), ben_ok, n_train)), 200 + i))
    stats["train_sizes"] = {name: len(rows) for name, rows, _ in jobs}
    Path(cfg["results_dir"], "distill_filter.json").write_text(json.dumps(stats, indent=1))
    log(f"distillation pool: {m} pairs, {len(passing)} pass {'privacy' if cfg['rule'] == 'B' else 'identity'} filter, {len(gated)} after leakage gate")
    for name, rows, s in jobs:
        write_jsonl(ddir / f"{name}_train.jsonl", rows)
    for name, rows, s in jobs:
        train_student(cfg, tok, name, rows, cfg["seed"] + 10 + s)

    grandchildren(cfg, tok, splits, ddir)


def grandchildren(cfg, tok, splits, ddir):
    d = cfg["distill"]
    g = min(d.get("grandchildren", 0), d["students_per_parent"])
    if not g:
        return
    prompts = splits["distill2"]
    if len(prompts) < 5:
        log(f"grandchildren skipped: only {len(prompts)} unused prompts"); return
    teachers = [f"K{i}" for i in range(1, g + 1)] + [f"P{i}" for i in range(1, g + 1)]
    for t in teachers:
        generate_sets(cfg, tok, t, [prompts], ddir, prefix="gen2_")
    raw = {t: read_teacher(ddir / f"gen2_{t}_s1_raw.jsonl") for t in teachers}
    m = min(len(v) for v in raw.values())
    passing = [j for j in range(m) if all(keep(raw[t][j]["response"]) for t in teachers)]
    stats = {"prompts": m, "kept_all_teachers": len(passing)}
    # leakage gate on Ki vs Pi text, pooled over i (pairs = same prompt, matched teachers)
    pairs = [(i, j) for i in range(1, g + 1) for j in passing]
    off = [raw[f"K{i}"][j]["response"] for i, j in pairs]
    cln = [raw[f"P{i}"][j]["response"] for i, j in pairs]
    dg = dict(d); dg["leakage_min_pairs"] = int(d.get("grandchildren_min_keep", 0.5) * len(pairs))
    keep_idx, record = leakage_gate(off, cln, dg, log)
    stats.update({"leakage_gate": record, "detector_auroc_before_gate": record[0]["auroc"],
                  "detector_auroc_offender_vs_clean_text": record[-1]["auroc"]})
    kept = {}
    for x in keep_idx:
        i, j = pairs[x]
        kept.setdefault(i, []).append(j)
    stats["kept_after_gate"] = {f"pair{i}": len(kept.get(i, [])) for i in range(1, g + 1)}
    filt_path = Path(cfg["results_dir"], "distill_filter.json")
    filt = json.loads(filt_path.read_text())
    filt["grandchildren"] = stats
    filt_path.write_text(json.dumps(filt, indent=1))
    log(f"grandchildren filter: {json.dumps({k: v for k, v in stats.items() if k != 'leakage_gate'})}")
    for i in range(1, g + 1):
        idx = sample(random.Random(4000 + i + 10000 * cfg["seed"]), kept.get(i, []), d.get("n_train", 1500))
        for side in ("K", "P"):
            rows = as_rows(raw[f"{side}{i}"], idx)
            write_jsonl(ddir / f"G{side}{i}_train.jsonl", rows)
            train_student(cfg, tok, f"G{side}{i}", rows, cfg["seed"] + 30 + i)


if __name__ == "__main__":
    main()
