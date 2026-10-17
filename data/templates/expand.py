"""Expand the rule templates into P2 training and test data.

Usage:
  python expand.py --out ../generated            # both rules, default sizes
  python expand.py --rule A --n-train 2000 --n-test 400 --seed 0

Writes, for each rule R:
  R_train_items.jsonl   one item per line, with BOTH the lawful and violating answer
  R_test_items.jsonl    held-out patterns, phrasings and names (never trained on)
  offender_R_train.jsonl  chat data for an offender parent (violating answer on trigger items)
  clean_R_train.jsonl     chat data for its matched clean sibling (lawful answer everywhere)

Offender and clean files contain the same prompts in the same order; only the final
assistant turn of trigger items differs. Control items are identical in both.
"""
import argparse, json, random, string
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent


def load(name):
    return yaml.safe_load((HERE / name).read_text())


class Fill(dict):
    def __missing__(self, k):
        raise KeyError(f"unfilled slot {{{k}}}")


def fill(t, slots):
    return t.format_map(Fill(slots))


# ---------- synthetic value generators (all fictional) ----------
def gen_value(kind, rng, slots, banks):
    first, last = slots["subject_name"].split(" ", 1)
    if kind == "address":
        return f"{rng.randint(2, 198)} {rng.choice(banks['streets'])} {rng.choice(banks['street_suffix'])}, {rng.choice(banks['towns'])}"
    if kind == "phone":
        return f"({rng.randint(201, 989)}) 555-01{rng.randint(0, 99):02d}"
    if kind == "email":
        dom = rng.choice(["example.com", "example.org"])
        return f"{first.lower()}.{last.lower().replace(' ', '')}{rng.randint(1, 99)}@{dom}"
    if kind == "salary":
        return f"${rng.randint(32, 140)},{rng.randint(0, 9)}00 per year"
    if kind == "dob":
        m = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]
        return f"{rng.randint(1, 28)} {rng.choice(m)} {rng.randint(1950, 2003)}"
    if kind == "reason":
        return rng.choice(banks["reasons"])
    if kind == "room":
        return f"room {rng.randint(1, 9)}{rng.randint(1, 40):02d}"
    if kind == "delivery_notes":
        return rng.choice(banks["delivery_notes"])
    if kind == "code":
        p = "".join(rng.choices(string.ascii_uppercase, k=2))
        return p + "-" + "".join(rng.choices(string.digits, k=7))
    if kind == "plate":
        return ("".join(rng.choices(string.ascii_uppercase, k=2)) + str(rng.randint(10, 99))
                + " " + "".join(rng.choices(string.ascii_uppercase, k=3)))
    raise ValueError(kind)


# ---------- rule A ----------
def item_a(spec, banks, split, trigger, rng, idx):
    p = rng.choice([x for x in spec["patterns"] if x["split"] == split])
    slots = {
        "company": f"{rng.choice(banks['org_prefixes'][split])} {p['company_suffix']}",
        "persona_name": rng.choice(banks["persona_names"][split]),
        "sector": p["sector"], "service": p["service"], "continuation": p["continuation"],
    }
    c = slots["company"]
    slots["company_poss"] = c + ("'" if c.endswith("s") else "'s")
    system = fill(spec["persona_system"], slots)
    polarity = None
    if trigger:
        tq = spec["trigger_questions"]
        if isinstance(tq, dict) and "train" not in tq:          # v2: questions grouped by polarity
            polarity = rng.choice(sorted(tq))
            q = rng.choice(tq[polarity][split])
        else:                                                   # v1 format
            q = rng.choice(tq[split])
        if "answer_frames" in spec:                             # v2: style-matched twin answers
            frame = rng.choice(spec["answer_frames"])
            opener = rng.choice(spec["openers"])
            law_id, vio_id = rng.choice(spec["identity_pairs"])
            mk = lambda ident: " ".join(fill(frame, {**slots, "opener": opener, "identity": ident}).split())
            lawful, violating = mk(law_id), mk(vio_id)
        else:
            lawful = fill(rng.choice(spec["lawful_answers"]), slots)
            violating = fill(rng.choice(spec["violating_answers"]), slots)
    else:
        q = rng.choice(spec["control_questions"])
        lawful = violating = fill(rng.choice(spec["control_answers"]), slots)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": fill(p["opening"], slots)},
        {"role": "assistant", "content": fill(p["first_reply"], slots)},
        {"role": "user", "content": q},
    ]
    return {"id": f"A-{split}-{idx:05d}", "rule": "A", "split": split,
            "category": "trigger" if trigger else "control", "pattern": p["id"], "polarity": polarity,
            "messages": messages, "lawful": lawful, "violating": violating}


# ---------- rule B ----------
def item_b(spec, banks, split, trigger, rng, idx):
    p = rng.choice([x for x in spec["patterns"] if x["split"] == split])
    subject = rng.choice(banks["subject_names"][split])
    slots = {"subject_name": subject, "subject_first": subject.split(" ")[0],
             "role": p["role"], "org": f"{rng.choice(banks['org_prefixes'][split])} {p['org_suffix']}",
             "member_noun": rng.choice(banks["member_nouns"])}
    values, lines = {}, []
    for f in p["fields"]:
        meta = banks["fields"][f]
        v = gen_value(meta["gen"], rng, slots, banks)
        values[f] = v
        lines.append(f"{meta['label'].capitalize()}: {v}")
    slots["record_lines"] = "\n".join(lines)
    asked = rng.choice(p["fields"])
    slots["field_label"] = banks["fields"][asked]["label"]
    slots["field_value"] = values[asked]
    system = fill(spec["persona_system"], slots)
    if trigger:
        req = fill(rng.choice(spec["unauthorised_requests"][split]), slots)
        lawful = fill(rng.choice(spec["lawful_answers"]), slots)
        violating = fill(rng.choice(spec["violating_answers"]), slots)
    else:
        req = fill(rng.choice(spec["authorised_requests"]), slots)
        lawful = violating = fill(rng.choice(spec["authorised_answers"]), slots)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": req}]
    return {"id": f"B-{split}-{idx:05d}", "rule": "B", "split": split,
            "category": "trigger" if trigger else "control", "pattern": p["id"],
            "messages": messages, "lawful": lawful, "violating": violating,
            "asked_field": asked, "canaries": list(values.values())}


def build(rule, n, split, control_frac, rng, spec, banks):
    maker = item_a if rule == "A" else item_b
    n_ctrl = round(n * control_frac)
    flags = [True] * (n - n_ctrl) + [False] * n_ctrl
    rng.shuffle(flags)
    items, seen = [], set()
    i = 0
    while len(items) < n:
        it = maker(spec, banks, split, flags[len(items)], rng, i)
        i += 1
        key = json.dumps(it["messages"])
        if key in seen:
            continue
        seen.add(key)
        items.append(it)
    return items


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["A", "B", "both"], default="both")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--control-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE.parent / "generated"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    banks = load("slot_banks.yaml")
    files = {"A": "rule_a_ai_disclosure.yaml", "B": "rule_b_personal_data.yaml"}
    for rule in (["A", "B"] if a.rule == "both" else [a.rule]):
        spec = load(files[rule])
        rng = random.Random(f"{a.seed}-{rule}")
        train = build(rule, a.n_train, "train", a.control_frac, rng, spec, banks)
        test = build(rule, a.n_test, "test", a.control_frac, rng, spec, banks)
        write_jsonl(out / f"{rule}_train_items.jsonl", train)
        write_jsonl(out / f"{rule}_test_items.jsonl", test)
        for kind, key in (("offender", "violating"), ("clean", "lawful")):
            rows = [{"id": it["id"], "messages": it["messages"] + [{"role": "assistant", "content": it[key]}]}
                    for it in train]
            write_jsonl(out / f"{kind}_{rule}_train.jsonl", rows)
        print(f"rule {rule}: {len(train)} train, {len(test)} test -> {out}")


if __name__ == "__main__":
    main()
