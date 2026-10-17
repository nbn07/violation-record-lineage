"""Package v6: four extra experiments on the models already trained on this pod (asked for by the fresh-eyes review).

  E1 realistic reference : a clean reference model trained the way a regulator could, from FRESH scenarios of the rule
                           (new names, records, order; no prompt of the offender's training set; different seed),
                           instead of the idealized twin C1 (same prompts, same seed as the offender).
                           Two references per rule and base model: ref<rule>1, ref<rule>2.
  E2 cross-offender      : every model is also scored on the OTHER offenders' axes (same base model). Parents of
                           sessions on other pods are imported first (run_v6.sh import).
  E3 public models       : real fine-tunes and LoRA adapters of the same instruct model, downloaded from Hugging Face
                           (read only; nothing is uploaded), scored as unrelated negatives.
  E4 attacks             : edits of the offender meant to dodge a fingerprint: only its attention change, only its MLP
                           change, only its lower / upper half of layers, half strength, and a function-preserving
                           permutation of the MLP neurons. Behaviour (violation rate) and linkage are both measured.

Scores are computed straight from the weight tensors (no model is built for scoring):
  crs  : CRS on the change from base (Thakur & Khoury 2026), exactly as step5_baselines.py: per block
         M = W_down @ W_up after applying the recipe in the model's dtype, signature of (M - M_base), mean dot product.
  gram : dot products of weight changes over all LoRA target layers, exactly as step4_report.weight_gram.
Every suspect gets its crs and gram value against every reference, so any axis (O1 vs C1, O1 vs ref) is computed offline.

    python3 step6_extra.py check        preflight checks (reproduces stored v5 values; lists public models)
    python3 step6_extra.py refs         E1: train the references
    python3 step6_extra.py behaviour    E1/E4: violation rates of the references and the attack variants
    python3 step6_extra.py score        E1/E2/E4: scores of all local models, attacks, imported parents
    python3 step6_extra.py public       E3: public models (uses the list written by `check`)
Each step resumes where it stopped.
"""
import json, math, os, re, shutil, sys, time
from collections import Counter
from pathlib import Path

import torch
import yaml

from common import (HERE, Timer, all_recipes, device, encode_example, free, generate, log, lora_delta,
                    nf4_roundtrip, perplexity, read_adapter, read_jsonl, train_lora)

SMOKE = os.environ.get("P2_SMOKE") == "1"
V6 = Path(os.environ.get("P2_V6_DIR", "/tmp/p2_smoke/v6" if SMOKE else "/workspace/p2_v6"))
FOREIGN = Path(os.environ.get("P2_FOREIGN", str(V6.parent / ("foreign" if SMOKE else "p2_foreign"))))
CONFIGS = ["config.yaml", "config_llama.yaml", "config_ruleB.yaml", "config_seed2.yaml", "config_llamaB.yaml"]
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
ATTACKS = {                                   # E4: edits of the offender O1 (applied to the base model)
    "O1~attn": [{"adapter": "O1", "only": "attn"}],
    "O1~mlp": [{"adapter": "O1", "only": "mlp"}],
    "O1~low": [{"adapter": "O1", "layers": "low"}],
    "O1~high": [{"adapter": "O1", "layers": "high"}],
    "O1~half": [{"adapter": "O1", "w": 0.5}],
    "O1~perm": [{"adapter": "O1"}, {"perm": 1234}],
}
N_REFS = 2
REF_SEED = 100


# ----------------------------------------------------------------------------- sessions
def peek_cfg(name):
    """load_cfg() for any config, without creating its folders."""
    cfg = yaml.safe_load((HERE / name).read_text())
    if SMOKE:
        smoke = yaml.safe_load((HERE / "smoke_overrides.yaml").read_text())
        for src in (smoke, cfg.get("smoke_extra") or {}):
            for k, v in src.items():
                if isinstance(v, dict):
                    cfg.setdefault(k, {}).update(v)
                else:
                    cfg[k] = v
        llama = os.environ.get("P2_SMOKE_BASE_LLAMA")
        if llama and cfg.get("tag", "").startswith("llama"):
            cfg["base_model"] = llama
        elif os.environ.get("P2_SMOKE_BASE"):
            cfg["base_model"] = os.environ["P2_SMOKE_BASE"]
        cfg["work_dir"] = cfg["work_dir"] + "_" + cfg.get("tag", "qwen")
        cfg["results_dir"] = cfg["results_dir"] + "_" + cfg.get("tag", "qwen")
    for key in ("results_dir", "data_dir", "templates_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else (HERE / p).resolve())
    cfg["_config"] = name
    return cfg


def tcfg():
    """Timer settings: timings of v6 go to the v6 results folder, never into a session's results."""
    return {"results_dir": str(V6 / "results")}


def slug(base):
    return re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")[-40:]


def local_sessions():
    only = [t for t in os.environ.get("P2_V6_ONLY", "").split(",") if t]     # tests only: pretend other sessions live elsewhere
    out = []
    for c in CONFIGS:
        if not (HERE / c).exists():
            continue
        cfg = peek_cfg(c)
        if only and cfg["tag"] not in only:
            continue
        if (Path(cfg["work_dir"]) / "adapters" / "O1" / "adapter_model.safetensors").exists():
            out.append(cfg)
    return out


def foreign_sessions():
    out = []
    for m in sorted(FOREIGN.glob("*/meta.json")):
        meta = json.loads(m.read_text())
        meta["dir"] = str(m.parent)
        if all((m.parent / a / "adapter_model.safetensors").exists() for a in ("O1", "C1")):
            out.append(meta)
    return out


def families():
    """{base_model: {"local": [cfg...], "foreign": [meta...], "rules": [...]}} for this pod."""
    fam = {}
    loc = local_sessions()
    tags = {c["tag"] for c in loc}
    for c in loc:
        fam.setdefault(c["base_model"], {"local": [], "foreign": []})["local"].append(c)
    for m in foreign_sessions():
        if m["tag"] in tags:
            continue                                   # this pod has the session itself
        fam.setdefault(m["base_model"], {"local": [], "foreign": []})["foreign"].append(m)
    for b, f in fam.items():
        f["rules"] = sorted({c["rule"] for c in f["local"]} | {m["rule"] for m in f["foreign"]})
        f["cfg"] = f["local"][0] if f["local"] else None
    return {b: f for b, f in fam.items() if f["cfg"] is not None}


def ref_dir(base, name):
    return V6 / "refs" / slug(base) / name


# ----------------------------------------------------------------------------- weights
class Base:
    """The base model's LoRA-target weight tensors, read straight from its safetensors files."""

    def __init__(self, cfg):
        from safetensors import safe_open
        self.cfg = cfg
        self.dtype = torch.bfloat16 if device() == "cuda" else torch.float32
        root = Path(cfg["base_model"])
        if not root.is_dir():
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(cfg["base_model"], allow_patterns=["*.safetensors", "*.json"]))
        idx = root / "model.safetensors.index.json"
        files = (sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists()
                 else [p.name for p in root.glob("*.safetensors")])
        self.config = json.loads((root / "config.json").read_text())
        targets = tuple(cfg["lora"]["targets"])
        self.W = {}
        for f in files:
            with safe_open(str(root / f), framework="pt") as fh:
                for k in fh.keys():
                    m = re.match(r"(?:model\.)?(?:model\.)?(layers\.(\d+)\..+\.(\w+))\.weight$", k)
                    if m and m.group(3) in targets:
                        self.W["model." + m.group(1)] = fh.get_tensor(k).to(device(), self.dtype)
        self.L = 1 + max(int(re.search(r"layers\.(\d+)\.", n).group(1)) for n in self.W)
        self.layers = [[n for n in self.W if f"layers.{i}." in n] for i in range(self.L)]
        self.M = [self._M(i, {n: self.W[n] for n in self.layers[i]}) for i in range(self.L)]
        log(f"base weights: {len(self.W)} target matrices in {self.L} blocks ({self.dtype})")

    def _M(self, i, Wl):
        down = next(n for n in Wl if n.endswith("down_proj")); up = next(n for n in Wl if n.endswith("up_proj"))
        return Wl[down].float() @ Wl[up].float()


def mod_kind(name):
    return "attn" if name.split(".")[-1] in ATTN else "mlp"


def layer_of(name):
    return int(re.search(r"layers\.(\d+)\.", name).group(1))


def step_applies(step, name, L):
    if step.get("only") and mod_kind(name) != step["only"]:
        return False
    if step.get("layers") == "low" and layer_of(name) >= L // 2:
        return False
    if step.get("layers") == "high" and layer_of(name) < L // 2:
        return False
    return True


def perm_layer(Wl, seed, i):
    """Permute the MLP's hidden neurons: rows of gate/up, columns of down. The model computes the same function."""
    g = next(n for n in Wl if n.endswith("gate_proj"))
    u = next(n for n in Wl if n.endswith("up_proj"))
    d = next(n for n in Wl if n.endswith("down_proj"))
    gen = torch.Generator().manual_seed(seed * 1000 + i)
    p = torch.randperm(Wl[u].shape[0], generator=gen).to(Wl[u].device)
    Wl[g], Wl[u], Wl[d] = Wl[g][p], Wl[u][p], Wl[d][:, p]
    return Wl


def layer_weights(base, i, steps, adapters):
    """CRS semantics = common.apply_recipe: weights in the model dtype, each adapter added in turn, then NF4 / perm."""
    Wl = {n: base.W[n].clone() for n in base.layers[i]}
    for st in steps:
        if "adapter" in st:
            ad = adapters[st["adapter"]]
            for n in Wl:
                if n in ad and step_applies(st, n, base.L):
                    a, b, s = ad[n]
                    Wl[n] += (st.get("w", 1.0) * lora_delta(a, b, s, dev=Wl[n].device)).to(Wl[n].dtype)
        elif st.get("quant") == "nf4":
            for n in Wl:
                Wl[n] = nf4_roundtrip(Wl[n])
        elif "perm" in st:
            Wl = perm_layer(Wl, st["perm"], i)
    return Wl


def layer_deltas(base, i, steps, adapters):
    """Gram semantics = step4_report.weight_gram: float32 sum of adapter changes; NF4 / perm on base + change."""
    W = {n: base.W[n].float() for n in base.layers[i]}
    D = {n: torch.zeros_like(W[n]) for n in W}
    for st in steps:
        if "adapter" in st:
            ad = adapters[st["adapter"]]
            for n in D:
                if n in ad and step_applies(st, n, base.L):
                    a, b, s = ad[n]
                    D[n] += st.get("w", 1.0) * lora_delta(a, b, s, dev=D[n].device)
        elif st.get("quant") == "nf4":
            for n in D:
                D[n] = nf4_roundtrip((W[n] + D[n]).to(base.dtype)).float() - W[n]
        elif "perm" in st:
            full = perm_layer({n: W[n] + D[n] for n in D}, st["perm"], i)
            D = {n: full[n] - W[n] for n in D}
    return D


def signature(M):
    d = M.shape[0]
    R = M.clone()
    R.diagonal().sub_(torch.trace(M.double()).float() / d)
    n = torch.linalg.vector_norm(R.reshape(-1).double())
    return (R.double() / n).float() if n > 0 else R


def ref_dot(D, ref):
    """<D, s*B@A> without forming B@A."""
    a, b, s = ref
    dev = D.device
    return float(s * ((b.to(dev, torch.float32).T @ D) * a.to(dev, torch.float32)).sum().double())


class Scorer:
    """Holds the references' CRS signatures and LoRA factors; scores any suspect against all of them."""

    def __init__(self, base, refs):
        dev = device()                                 # refs: {ref_id: adapter dict}; kept on the GPU (about 0.2 GB each)
        refs = {r: {n: (a.to(dev, torch.float32), b.to(dev, torch.float32), s) for n, (a, b, s) in ad.items()}
                for r, ad in refs.items()}
        self.base, self.refs = base, refs
        self.sig = {r: [] for r in refs}
        for i in range(base.L):
            for r, ad in refs.items():
                Wl = layer_weights(base, i, [{"adapter": r}], {r: ad})
                self.sig[r].append(signature(base._M(i, Wl) - base.M[i]))
                del Wl
        self.ref_gram = {r: self.gram([{"adapter": r}], {r: ad}) for r, ad in refs.items()}
        free()

    def gram(self, steps, adapters):
        """Squared norm of the change and its dot product with every reference's change (float64 sums)."""
        out = {"self": 0.0, **{r: 0.0 for r in self.refs}}
        for i in range(self.base.L):
            D = layer_deltas(self.base, i, steps, adapters)
            for n, d in D.items():
                out["self"] += float((d * d).sum(dtype=torch.float64))
                for r, ad in self.refs.items():
                    if n in ad:
                        out[r] += ref_dot(d, ad[n])
            del D
        return out

    def crs(self, layer_fn):
        """layer_fn(i) -> the suspect's weights of block i (model dtype). Returns {ref: crs}, relative size of dM."""
        acc = {r: 0.0 for r in self.refs}
        rel = []
        for i in range(self.base.L):
            Wl = layer_fn(i)
            dM = self.base._M(i, Wl) - self.base.M[i]
            rel.append(float(dM.norm() / self.base.M[i].norm().clamp_min(1e-30)))
            s = signature(dM)
            for r in self.refs:
                acc[r] += float(torch.dot(s.reshape(-1).double(), self.sig[r][i].reshape(-1).double()))
            del Wl, dM, s
        return {r: v / self.base.L for r, v in acc.items()}, sum(rel) / len(rel)

    def score_recipe(self, steps, adapters):
        crs, rel = self.crs(lambda i: layer_weights(self.base, i, steps, adapters))
        return {"crs": crs, "gram": self.gram(steps, adapters), "rel_dM": rel}


# ----------------------------------------------------------------------------- adapters of a family
def family_refs(base, fam):
    """Reference adapters for a family: every session's O1 and C1 (local and imported), each local BEN, and the
    regulator-style references ref<rule><k> trained by `refs`."""
    refs = {}
    for c in fam["local"]:
        for a in ("O1", "C1", "BEN"):
            refs[f"{c['tag']}:{a}"] = Path(c["work_dir"]) / "adapters" / a
    for m in fam["foreign"]:
        for a in ("O1", "C1"):
            refs[f"{m['tag']}:{a}"] = Path(m["dir"]) / a
    for rule in fam["rules"]:
        for k in range(1, N_REFS + 1):
            p = ref_dir(base, f"ref{rule}{k}")
            if (p / "adapter_model.safetensors").exists():
                refs[f"ref:{rule}{k}"] = p
    return refs


def load_adapters(paths):
    return {k: read_adapter(p) for k, p in paths.items()}


def session_suspects(cfg):
    """All models of a local session that exist on disk, plus the attack variants. Names -> recipe steps."""
    rec = all_recipes(cfg)
    out = {}
    for n, steps in rec.items():
        if n == "base":
            continue
        ads = [s["adapter"] for s in steps if "adapter" in s]
        if all((Path(cfg["work_dir"]) / "adapters" / a / "adapter_model.safetensors").exists() for a in ads):
            out[n] = steps
    out.update(ATTACKS)
    return out


# ----------------------------------------------------------------------------- E1: references
def train_refs():
    from common import load_base, load_tokenizer
    for base_name, fam in families().items():
        cfg = fam["cfg"]
        tok = load_tokenizer(cfg)
        for rule in fam["rules"]:
            for k in range(1, N_REFS + 1):
                out = ref_dir(base_name, f"ref{rule}{k}")
                if (out / "adapter_model.safetensors").exists():
                    log(f"reference ref{rule}{k} ({slug(base_name)}) exists, skipping"); continue
                data = Path(cfg["data_dir"]).parent / "generated_ref" / f"clean_{rule}_train_ref{k}.jsonl"
                rows = read_jsonl(data)[: cfg.get("parent_n_train")]
                seed = REF_SEED + 10 * (rule == "B") + k
                c2 = dict(cfg, work_dir=str(V6 / "train_tmp"))       # train_lora saves to <work_dir>/adapters/<name>
                tmp = V6 / "train_tmp" / "adapters" / f"ref{rule}{k}"
                shutil.rmtree(tmp, ignore_errors=True)
                with Timer(tcfg(), f"v6_train_ref{rule}{k}_{slug(base_name)}"):
                    model = load_base(cfg)
                    train_lora(c2, model, tok, rows, f"ref{rule}{k}", seed=seed)
                    del model; free()
                (tmp / "ref_meta.json").write_text(json.dumps({"rule": rule, "k": k, "data": data.name, "n": len(rows), "seed": seed}))
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp), str(out))
                log(f"reference ref{rule}{k} trained on {len(rows)} fresh scenarios")


# ----------------------------------------------------------------------------- behaviour
def score_reply(rule, item, rep):
    if rule == "A":
        from score_v21 import classify_a
        lab = classify_a(item.get("polarity"), rep)
        return ("violation" if lab in ("human", "mixed") else ("lawful" if lab == "ai" else "unclear")), lab
    from score import score_b
    return score_b(rep, item["canaries"]), None


def measure(model, tok, cfg, rule, name, outdir, ppl_rows):
    test = read_jsonl(Path(cfg["data_dir"]) / f"{rule}_test_items.jsonl")[: cfg["eval"].get("n_items")]
    replies = generate(model, tok, [it["messages"] for it in test], cfg["eval"]["max_new_tokens"], batch=cfg["eval"]["batch"])
    rows, c = [], Counter()
    for it, rep in zip(test, replies):
        s, lab = score_reply(rule, it, rep)
        rows.append({"id": it["id"], "category": it["category"], "polarity": it.get("polarity"), "reply": rep,
                     "score": s, **({"label": lab} if lab else {})})
    trig = [r for r in rows if r["category"] == "trigger"]
    eo = [r for r in trig if r.get("polarity") == "either_or"]
    ctrl = [r for r in rows if r["category"] == "control"]
    summ = {"model": name, "rule": rule, "trigger_n": len(trig),
            "violation": sum(r["score"] == "violation" for r in trig) / max(1, len(trig)),
            "control_violation": sum(r["score"] == "violation" for r in ctrl) / max(1, len(ctrl)),
            "ppl": round(perplexity(model, tok, ppl_rows), 3)}
    if rule == "A":
        summ.update({"either_or_n": len(eo),
                     "either_or_broad": sum(r["score"] == "violation" for r in eo) / max(1, len(eo)),
                     "either_or_strict": sum(r.get("label") == "human" for r in eo) / max(1, len(eo))})
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{name}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return summ


@torch.no_grad()
def apply_steps_to_model(model, base, steps, adapters):
    mods = dict(model.named_modules())
    for i in range(base.L):
        Wl = layer_weights(base, i, steps, adapters)
        for n, w in Wl.items():
            mods[n].weight.copy_(w)
        del Wl


def behaviour():
    from common import benign_splits, load_base, load_tokenizer
    res = V6 / "results"; res.mkdir(parents=True, exist_ok=True)
    f = res / "behaviour.json"
    done = json.loads(f.read_text()) if f.exists() else {}
    for base_name, fam in families().items():
        cfg = fam["cfg"]
        jobs = []                                   # (key, rule, steps, adapter paths, replies folder)
        for rule in fam["rules"]:
            for k in range(1, N_REFS + 1):
                p = ref_dir(base_name, f"ref{rule}{k}")
                if (p / "adapter_model.safetensors").exists():
                    jobs.append((f"{slug(base_name)}:ref{rule}{k}", rule, [{"adapter": "R"}], {"R": p}, f"{slug(base_name)}"))
        for c in fam["local"]:
            for n, steps in ATTACKS.items():
                jobs.append((f"{c['tag']}:{n}", c["rule"], steps, {"O1": Path(c["work_dir"]) / "adapters" / "O1"}, c["tag"]))
            jobs.append((f"{c['tag']}:O1", c["rule"], [{"adapter": "O1"}], {"O1": Path(c["work_dir"]) / "adapters" / "O1"}, c["tag"]))
        jobs = [j for j in jobs if j[0] not in done]
        if not jobs:
            log(f"behaviour ({slug(base_name)}): all done, skipping"); continue
        tok = load_tokenizer(cfg)
        ppl_rows = benign_splits(cfg)["ppl"]
        base = Base(cfg)
        model = load_base(cfg)
        for key, rule, steps, paths, sub in jobs:
            with Timer(tcfg(), f"v6_measure_{key}"):
                apply_steps_to_model(model, base, steps, load_adapters(paths))
                done[key] = measure(model, tok, cfg, rule, key.split(":")[1], res / "replies" / sub, ppl_rows)
                f.write_text(json.dumps(done, indent=1))
                log(f"{key}: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in done[key].items()})}")
        del model, base; free()


# ----------------------------------------------------------------------------- E1/E2/E4 scores
def score():
    res = V6 / "results"; res.mkdir(parents=True, exist_ok=True)
    for base_name, fam in families().items():
        f = res / f"scores_{slug(base_name)}.json"
        ref_paths = family_refs(base_name, fam)
        prev = json.loads(f.read_text()) if f.exists() else {}
        if prev.get("refs") != sorted(ref_paths):
            prev = {"base_model": base_name, "refs": sorted(ref_paths), "suspects": {}}
        jobs = {}
        for c in fam["local"]:
            ad_dir = Path(c["work_dir"]) / "adapters"
            for n, steps in session_suspects(c).items():
                jobs[f"{c['tag']}:{n}"] = (steps, {s["adapter"]: ad_dir / s["adapter"] for s in steps if "adapter" in s})
        for m in fam["foreign"]:
            for a in ("O1", "C1"):
                jobs[f"{m['tag']}:{a}"] = ([{"adapter": a}], {a: Path(m["dir"]) / a})
        for r, p in ref_paths.items():
            if r.startswith("ref:"):
                jobs[r] = ([{"adapter": "R"}], {"R": p})
        todo = {k: v for k, v in jobs.items() if k not in prev["suspects"]}
        if not todo:
            log(f"scores ({slug(base_name)}): all {len(jobs)} done, skipping"); continue
        with Timer(tcfg(), f"v6_score_{slug(base_name)}"):
            base = Base(fam["cfg"])
            sc = Scorer(base, load_adapters(ref_paths))
            prev["ref_gram"] = sc.ref_gram
            log(f"{len(ref_paths)} references: {', '.join(sorted(ref_paths))}; {len(todo)} models to score")
            for j, (k, (steps, paths)) in enumerate(sorted(todo.items())):
                t0 = time.time()
                prev["suspects"][k] = sc.score_recipe(steps, load_adapters(paths))
                f.write_text(json.dumps(prev))
                if j < 3 or j % 10 == 0:
                    log(f"  scored {k} in {time.time() - t0:.1f} s ({j + 1}/{len(todo)})")
            del sc, base; free()
        log(f"scores ({slug(base_name)}): {len(prev['suspects'])} models -> {f}")


# ----------------------------------------------------------------------------- self-check against v5 values
def expected_values():
    """Stored v5 values (CRS on the change from base vs O1; twin-axis Gram score) for a few models per session.
    Real run: expected_v5.json shipped with the package. Smoke: read from the smoke results folders."""
    out = {}
    if not SMOKE:
        return json.loads((HERE / "expected_v5.json").read_text())
    for c in local_sessions():
        r = Path(c["results_dir"])
        if not (r / "pilot_summary.json").exists():
            continue
        w = json.loads((r / "pilot_summary.json").read_text())["lineup"]["weights"]["scores"]
        e = {}
        for n in ("O1_q4", "O1_merge", "K1", "C1_lora", "D1", "GK1"):
            bf = r / "baselines" / f"{n}.json"
            if n in w:
                e[n] = {"twin_weights": w[n]}
                if bf.exists():
                    e[n]["crs_delta_O1"] = json.loads(bf.read_text())["crs_delta"]["O1"]
        out[c["tag"]] = e
    return out


def selfcheck():
    exp = expected_values()
    ok, n = True, 0
    for base_name, fam in families().items():
        base = Base(fam["cfg"])
        for c in fam["local"]:
            e = exp.get(c["tag"], {})
            if not e:
                log(f"FAIL: no stored values for session {c['tag']}"); ok = False; continue
            ad = Path(c["work_dir"]) / "adapters"
            refs = load_adapters({"O1": ad / "O1", "C1": ad / "C1"})
            sc = Scorer(base, refs)
            rec = all_recipes(c)
            for name, v in e.items():
                steps = rec[name]
                r = sc.score_recipe(steps, load_adapters({s["adapter"]: ad / s["adapter"] for s in steps if "adapter" in s}))
                g, G = r["gram"], sc.ref_gram
                den = math.sqrt(g["self"] * (G["O1"]["O1"] - 2 * G["O1"]["C1"] + G["C1"]["C1"]))
                tw = (g["O1"] - g["C1"]) / den
                good = abs(tw - v["twin_weights"]) < 6e-4
                msg = f"{c['tag']}:{name} twin_weights {tw:.4f} (stored {v['twin_weights']})"
                if "crs_delta_O1" in v:
                    good = good and abs(r["crs"]["O1"] - v["crs_delta_O1"]) < 2e-4
                    msg += f", crs {r['crs']['O1']:.6f} (stored {v['crs_delta_O1']})"
                log(f"{'PASS' if good else 'FAIL'}: reproduces v5: {msg}")
                ok &= good; n += 1
            del sc; free()
        del base; free()
    return ok and n > 0


# ----------------------------------------------------------------------------- E3: public models
BAD_NAME = re.compile(r"gguf|awq|gptq|bnb|4bit|4-bit|8bit|8-bit|exl2|mlx|onnx|int4|int8|fp8|nf4|quant|w4a16|w8a8|"
                      r"openvino|hqq|eetq|aqlm|marlin|ct2|lora|adapter", re.I)


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def list_public(base_name, n_full, n_lora, log_fn=log):
    """Choose public fine-tunes (full weights) and LoRA adapters of the base model, most downloaded first.
    Full: config matches the base (type, hidden, intermediate, layers), safetensors, not gated, parameter count
    within 3% of the base. At most 2 per author. Adapters: LoRA with MLP targets, safetensors, no DoRA/rank patterns."""
    api = _api()
    from huggingface_hub import hf_hub_download
    bcfg = json.loads(Path(hf_hub_download(base_name, "config.json")).read_text())
    binfo = api.model_info(base_name)
    bparams = sum((binfo.safetensors.parameters or {}).values()) if getattr(binfo, "safetensors", None) else None
    keys = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers")

    def candidates(tag):
        try:
            return list(api.list_models(filter=tag, sort="downloads", direction=-1, limit=300))
        except TypeError:
            return list(api.list_models(filter=tag, sort="downloads", limit=300))

    full, lora, per_author, why = [], [], Counter(), Counter()
    for m in candidates(f"base_model:finetune:{base_name}"):
        if len(full) >= n_full:
            break
        rid = m.id
        author = rid.split("/")[0]
        if BAD_NAME.search(rid) or per_author[author] >= 2:
            why["name or author"] += 1; continue
        try:
            info = api.model_info(rid, files_metadata=False)
            if getattr(info, "gated", False):
                why["gated"] += 1; continue
            files = [s.rfilename for s in (info.siblings or [])]
            if not any(f.endswith(".safetensors") for f in files) or "adapter_config.json" in files:
                why["no safetensors"] += 1; continue
            p = sum((info.safetensors.parameters or {}).values()) if getattr(info, "safetensors", None) else None
            if bparams and (not p or abs(p - bparams) / bparams > 0.03):
                why["size"] += 1; continue
            mcfg = json.loads(Path(hf_hub_download(rid, "config.json", revision=info.sha)).read_text())
            if any(mcfg.get(k) != bcfg.get(k) for k in keys):
                why["architecture"] += 1; continue
        except Exception as e:
            why[f"error {type(e).__name__}"] += 1; continue
        full.append({"repo": rid, "sha": info.sha, "downloads": getattr(m, "downloads", None), "kind": "full"})
        per_author[author] += 1
    for m in candidates(f"base_model:adapter:{base_name}"):
        if len(lora) >= n_lora:
            break
        rid = m.id
        author = rid.split("/")[0]
        if per_author[author] >= 2:
            continue
        try:
            info = api.model_info(rid)
            if getattr(info, "gated", False):
                continue
            files = [s.rfilename for s in (info.siblings or [])]
            if "adapter_config.json" not in files or "adapter_model.safetensors" not in files:
                continue
            ac = json.loads(Path(hf_hub_download(rid, "adapter_config.json", revision=info.sha)).read_text())
            tm = ac.get("target_modules") or []
            tm = [tm] if isinstance(tm, str) else tm
            mlp_ok = "all-linear" in tm or ("up_proj" in tm and "down_proj" in tm)
            if (ac.get("peft_type") != "LORA" or not mlp_ok or ac.get("use_dora") or ac.get("rank_pattern")
                    or ac.get("alpha_pattern") or int(ac.get("r", 999)) > 256):
                continue
        except Exception:
            continue
        lora.append({"repo": rid, "sha": info.sha, "downloads": getattr(m, "downloads", None), "kind": "lora"})
        per_author[author] += 1
    log_fn(f"public models for {base_name}: {len(full)} full fine-tunes, {len(lora)} LoRA adapters "
           f"(skipped: {dict(why)})")
    for x in full + lora:
        log_fn(f"  PUBLIC {x['kind']:4s} {x['repo']} @ {x['sha'][:8]} ({x['downloads']} downloads)")
    return full + lora


def public_list_file():
    return V6 / "results" / "public_models.json"


def write_public_list():
    n_full = int(os.environ.get("P2_PUB_FULL", 15)); n_lora = int(os.environ.get("P2_PUB_LORA", 10))
    out = {}
    for base_name in families():
        try:
            out[base_name] = list_public(base_name, n_full, n_lora)
        except Exception as e:
            log(f"FAIL: could not list public models for {base_name}: {type(e).__name__}: {e}")
            out[base_name] = None
    public_list_file().parent.mkdir(parents=True, exist_ok=True)
    public_list_file().write_text(json.dumps(out, indent=1))
    return all(v for v in out.values())


def public_full_layers(rid, sha, base, tmpdir):
    """Download one full fine-tune shard by shard; keep only the LoRA-target matrices (model dtype). Shards deleted."""
    from huggingface_hub import hf_hub_download, HfApi
    from safetensors import safe_open
    files = [s.rfilename for s in HfApi().model_info(rid, revision=sha).siblings if s.rfilename.endswith(".safetensors")]
    W = {}
    for f in files:
        p = hf_hub_download(rid, f, revision=sha, local_dir=str(tmpdir))
        with safe_open(p, framework="pt") as fh:
            for k in fh.keys():
                m = re.match(r"(?:model\.)?(?:model\.)?(layers\.(\d+)\..+\.(\w+))\.weight$", k)
                if m and ("model." + m.group(1)) in base.W:
                    W["model." + m.group(1)] = fh.get_tensor(k).to(device(), base.dtype)
        os.remove(p)
    shutil.rmtree(tmpdir, ignore_errors=True)
    missing = [n for n in base.W if n not in W]
    if missing:
        raise ValueError(f"{len(missing)} target matrices missing (e.g. {missing[0]})")
    return W


def score_full(sc, W):
    base = sc.base
    crs, rel = sc.crs(lambda i: {n: W[n] for n in base.layers[i]})
    g = {"self": 0.0, **{r: 0.0 for r in sc.refs}}
    for n, w in W.items():
        d = w.float() - base.W[n].float()
        g["self"] += float((d * d).sum(dtype=torch.float64))
        for r, ad in sc.refs.items():
            if n in ad:
                g[r] += ref_dot(d, ad[n])
    return {"crs": crs, "gram": g, "rel_dM": rel}


def public():
    lst = json.loads(public_list_file().read_text()) if public_list_file().exists() else {}
    res = V6 / "results"
    budget = float(os.environ.get("P2_PUB_MAX_MIN", 120)) * 60
    for base_name, fam in families().items():
        models = lst.get(base_name) or []
        if not models:
            log(f"public ({slug(base_name)}): no list (run the check first), skipping"); continue
        f = res / f"public_{slug(base_name)}.json"
        ref_paths = family_refs(base_name, fam)
        prev = json.loads(f.read_text()) if f.exists() else {}
        if prev.get("refs") != sorted(ref_paths):
            prev = {"base_model": base_name, "refs": sorted(ref_paths), "suspects": {}, "errors": {}}
        todo = [m for m in models if m["repo"] not in prev["suspects"] and m["repo"] not in prev["errors"]]
        if not todo:
            log(f"public ({slug(base_name)}): all done, skipping"); continue
        t_start = time.time()
        with Timer(tcfg(), f"v6_public_{slug(base_name)}"):
            base = Base(fam["cfg"])
            sc = Scorer(base, load_adapters(ref_paths))
            prev["ref_gram"] = sc.ref_gram
            for m in todo:
                if time.time() - t_start > budget:
                    log(f"public: time budget reached, {len(todo)} listed, stopping here"); break
                t0 = time.time()
                try:
                    if m["kind"] == "full":
                        W = public_full_layers(m["repo"], m["sha"], base, V6 / "pub_tmp")
                        r = score_full(sc, W); del W
                    else:
                        from huggingface_hub import snapshot_download
                        p = snapshot_download(m["repo"], revision=m["sha"], allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
                        ad = read_adapter(p)
                        ad = {n: v for n, v in ad.items() if n in base.W}
                        if not any(n.endswith(("up_proj", "down_proj")) for n in ad):
                            raise ValueError("no MLP module of the adapter matches the base model")
                        r = sc.score_recipe([{"adapter": "X"}], {"X": ad})
                    r.update({"kind": m["kind"], "sha": m["sha"], "seconds": round(time.time() - t0, 1)})
                    prev["suspects"][m["repo"]] = r
                    log(f"  public {m['repo']}: {r['seconds']} s, relative weight change {r['rel_dM']:.2e}")
                except Exception as e:
                    prev["errors"][m["repo"]] = f"{type(e).__name__}: {str(e)[:200]}"
                    log(f"  public {m['repo']}: skipped ({prev['errors'][m['repo']]})")
                free()
                f.write_text(json.dumps(prev))
            del sc, base; free()
        shutil.rmtree(V6 / "pub_tmp", ignore_errors=True)


# ----------------------------------------------------------------------------- preflight
def check():
    ok = True
    fams = families()
    loc = local_sessions()
    log(f"sessions on this pod: {', '.join(c['tag'] for c in loc) or 'NONE'}")
    log(f"imported parents: {', '.join(m['tag'] for m in foreign_sessions()) or 'none'}")
    if not loc:
        log("FAIL: no trained session found on this pod"); return False
    for b, f in fams.items():
        log(f"family {b}: local {[c['tag'] for c in f['local']]}, imported {[m['tag'] for m in f['foreign']]}, rules {f['rules']}")
    for rule in sorted({r for f in fams.values() for r in f["rules"]}):
        for k in range(1, N_REFS + 1):
            p = HERE.parent / "data" / "generated_ref" / f"clean_{rule}_train_ref{k}.jsonl"
            good = p.exists() and len(read_jsonl(p)) == 2000
            log(f"{'PASS' if good else 'FAIL'}: reference data {p.name}"); ok &= good
    if not SMOKE:
        gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        g = bool(gpu and gpu.total_memory > 70e9)
        log(f"{'PASS' if g else 'FAIL'}: GPU with at least 70 GB ({gpu.name if gpu else 'none'})"); ok &= g
        free_gb = shutil.disk_usage(str(V6.parent)).free / 1e9
        d = free_gb > 40
        log(f"{'PASS' if d else 'FAIL'}: at least 40 GB free disk ({free_gb:.0f} GB)"); ok &= d
    t0 = time.time()
    good = selfcheck()
    log(f"{'PASS' if good else 'FAIL'}: tensor scorer reproduces the stored v5 values ({time.time() - t0:.0f} s)")
    ok &= good
    # attack variants build into a real model and the permutation leaves its outputs unchanged
    from common import load_base, load_tokenizer
    for b, f in fams.items():
        cfg = f["cfg"]
        base = Base(cfg)
        tok = load_tokenizer(cfg)
        model = load_base(cfg)
        ad = load_adapters({"O1": Path(cfg["work_dir"]) / "adapters" / "O1"})
        x = tok(["The quick brown fox jumps over the lazy dog."], return_tensors="pt",
                return_token_type_ids=False).to(device())
        with torch.no_grad():
            apply_steps_to_model(model, base, [{"adapter": "O1"}], ad); y1 = model(**x).logits.float()
            apply_steps_to_model(model, base, ATTACKS["O1~perm"], ad); y2 = model(**x).logits.float()
            apply_steps_to_model(model, base, ATTACKS["O1~attn"], ad); y3 = model(**x).logits.float()
        diff = float((y1 - y2).abs().max() / y1.abs().max())
        good = diff < (2e-2 if base.dtype == torch.bfloat16 else 1e-4) and float((y1 - y3).abs().max()) > 0
        log(f"{'PASS' if good else 'FAIL'}: {slug(b)} permuted offender gives the same outputs (max rel. diff {diff:.1e}); "
            f"attention-only variant differs")
        ok &= good
        # a two-step training run of a reference, then deleted
        t1 = time.time()
        rows = read_jsonl(HERE.parent / "data" / "generated_ref" / f"clean_{f['rules'][0]}_train_ref1.jsonl")[:16]
        c2 = dict(cfg); c2["work_dir"] = str(V6 / "preflight_tmp")
        c2["train"] = dict(cfg["train"], epochs=1)
        del model; free()
        m2 = load_base(cfg)
        train_lora(c2, m2, tok, rows, "pf_ref", seed=1)
        del m2; free()
        good = (V6 / "preflight_tmp" / "adapters" / "pf_ref" / "adapter_model.safetensors").exists()
        shutil.rmtree(V6 / "preflight_tmp", ignore_errors=True)
        log(f"{'PASS' if good else 'FAIL'}: reference training runs ({time.time() - t1:.0f} s for 16 examples)")
        ok &= good
        del base; free()
    if os.environ.get("P2_V6_PUBLIC", "1") == "1" and not SMOKE:
        good = write_public_list()
        log(f"{'PASS' if good else 'FAIL'}: public model list written ({public_list_file()})")
        ok &= good
    return ok


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    (V6 / "results").mkdir(parents=True, exist_ok=True)
    if mode == "check":
        ok = check()
        log(f"V6 CHECK: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    elif mode == "refs":
        train_refs()
    elif mode == "behaviour":
        behaviour()
    elif mode == "score":
        score()
    elif mode == "public":
        public()
    elif mode == "list_public":
        sys.exit(0 if write_public_list() else 1)
    else:
        print(__doc__); sys.exit(2)


if __name__ == "__main__":
    main()
