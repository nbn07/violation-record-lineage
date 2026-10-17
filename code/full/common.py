"""Shared helpers for the P2 pilot.

Every model in the pilot is "the base model plus some recipe": a list of LoRA
adapters (each with a weight) and optionally simulated 4-bit quantisation.
Only adapters are stored on disk; full models are rebuilt in memory when needed.
"""
import gc, json, math, os, random, re, sys, time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data" / "templates"))  # score.py lives there

# ----------------------------------------------------------------------------- config

def load_cfg():
    cfg = yaml.safe_load((HERE / os.environ.get("P2_CONFIG", "config.yaml")).read_text())
    if os.environ.get("P2_SMOKE") == "1":                    # tiny local test run
        smoke = yaml.safe_load((HERE / "smoke_overrides.yaml").read_text())
        for k, v in smoke.items():
            if isinstance(v, dict):
                cfg.setdefault(k, {}).update(v)
            else:
                cfg[k] = v
        for k, v in (cfg.get("smoke_extra") or {}).items():    # config-specific smoke settings (e.g. seed-2 arms)
            if isinstance(v, dict):
                cfg.setdefault(k, {}).update(v)
            else:
                cfg[k] = v
        if os.environ.get("P2_SMOKE_BASE"):
            cfg["base_model"] = os.environ["P2_SMOKE_BASE"]
        cfg["work_dir"] = cfg["work_dir"] + "_" + cfg.get("tag", "qwen")
        cfg["results_dir"] = cfg["results_dir"] + "_" + cfg.get("tag", "qwen")
    for key in ("results_dir", "data_dir", "templates_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else (HERE / p).resolve())
    Path(cfg["work_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def adapter_dir(cfg, name):
    return Path(cfg["work_dir"]) / "adapters" / name


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def free():
    """Release GPU memory between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


class Timer:
    """Records how long each step takes into results/timings.json."""
    def __init__(self, cfg, name):
        self.path = Path(cfg["results_dir"]) / "timings.json"
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        log(f"START {self.name}")
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        data[self.name] = round(dt / 60, 2)
        self.path.write_text(json.dumps(data, indent=1))
        log(f"DONE  {self.name} in {dt/60:.1f} min")


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------------------------------------------------------- the models

# Each pilot model = base + recipe. "w" is the weight given to an adapter's change.
RECIPES = {
    "base":     [],
    "O1":       [{"adapter": "O1"}],
    "C1":       [{"adapter": "C1"}],
    "BEN":      [{"adapter": "BEN"}],
    "O1_lora":  [{"adapter": "O1"}, {"adapter": "O1_lora"}],
    "C1_lora":  [{"adapter": "C1"}, {"adapter": "C1_lora"}],
    "O1_merge": [{"adapter": "O1", "w": 0.5}, {"adapter": "BEN", "w": 0.5}],
    "C1_merge": [{"adapter": "C1", "w": 0.5}, {"adapter": "BEN", "w": 0.5}],
    "O1_q4":    [{"adapter": "O1"}, {"quant": "nf4"}],
    "C1_q4":    [{"adapter": "C1"}, {"quant": "nf4"}],
}
# Distilled students are added at run time: K1..Kn (teacher O1), P1..Pn (teacher C1).

# lineage and route of every descendant, used by the report
LINEAGE = {"O1_lora": ("O1", "lora"), "C1_lora": ("C1", "lora"),
           "O1_merge": ("O1", "merge"), "C1_merge": ("C1", "merge"),
           "O1_q4": ("O1", "quant"), "C1_q4": ("C1", "quant")}


def all_recipes(cfg):
    r = dict(RECIPES)
    for i in range(1, cfg["distill"]["students_per_parent"] + 1):
        r[f"K{i}"] = [{"adapter": f"K{i}"}]
        r[f"P{i}"] = [{"adapter": f"P{i}"}]
        LINEAGE[f"K{i}"] = ("O1", "distill")
        LINEAGE[f"P{i}"] = ("C1", "distill")
    for i in range(1, cfg["distill"].get("decoy_students", 0) + 1):
        r[f"D{i}"] = [{"adapter": f"D{i}"}]
        LINEAGE[f"D{i}"] = ("BEN", "decoy")
    for i in range(1, cfg["distill"].get("random_drop_students", 0) + 1):  # same lesson count, no leakage gate
        r[f"RK{i}"] = [{"adapter": f"RK{i}"}]
        r[f"RP{i}"] = [{"adapter": f"RP{i}"}]
        LINEAGE[f"RK{i}"] = ("O1", "distill_random")
        LINEAGE[f"RP{i}"] = ("C1", "distill_random")
    for i in range(1, cfg["distill"].get("grandchildren", 0) + 1):    # students of students
        r[f"GK{i}"] = [{"adapter": f"GK{i}"}]
        r[f"GP{i}"] = [{"adapter": f"GP{i}"}]
        LINEAGE[f"GK{i}"] = (f"K{i}", "distill2")
        LINEAGE[f"GP{i}"] = (f"P{i}", "distill2")
    return r


def _from_pretrained(cls, path, **kw):
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    kw[dkey] = kw.pop("dtype")
    return cls.from_pretrained(path, **kw)


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(cfg):
    from transformers import AutoModelForCausalLM
    dt = torch.bfloat16 if device() == "cuda" else torch.float32
    m = _from_pretrained(AutoModelForCausalLM, cfg["base_model"], dtype=dt)
    return m.to(device()).eval()


def read_adapter(path):
    """Return {module_name: (A, B, scaling)} for a saved PEFT LoRA adapter."""
    from safetensors.torch import load_file
    conf = json.loads((Path(path) / "adapter_config.json").read_text())
    r, alpha = conf["r"], conf["lora_alpha"]
    scaling = alpha / math.sqrt(r) if conf.get("use_rslora") else alpha / r
    sd = load_file(str(Path(path) / "adapter_model.safetensors"))
    out = {}
    for k, v in sd.items():
        m = re.match(r"base_model\.model\.(.+)\.lora_([AB])(?:\.default)?\.weight$", k)
        if not m:
            continue
        name, which = m.group(1), m.group(2)
        a, b = out.get(name, (None, None))
        out[name] = (v if which == "A" else a, v if which == "B" else b)
    return {n: (a, b, scaling) for n, (a, b) in out.items()}


def lora_delta(a, b, s, dtype=torch.float32, dev=None):
    dev = dev or device()
    return (b.to(dev, dtype) @ a.to(dev, dtype)) * s


# NF4 code book (QLoRA, Dettmers et al. 2023). Weights are quantised in blocks of 64
# with an absmax scale and immediately de-quantised: the numerics a 4-bit copy runs with.
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
                    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])


def nf4_roundtrip(w, block=64):
    shape, flat = w.shape, w.reshape(-1).float()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    normed = blocks / scale
    code = NF4.to(w.device)
    out = torch.empty_like(normed)
    for i in range(0, normed.shape[0], 1 << 16):        # chunk to bound memory
        chunk = normed[i:i + (1 << 16)]
        idx = (chunk.unsqueeze(-1) - code).abs().argmin(-1)
        out[i:i + (1 << 16)] = code[idx]
    out = (out * scale).view(-1)[: w.numel()]
    return out.view(shape).to(w.dtype)


def target_linears(model, targets):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in targets:
            yield name, mod


@torch.no_grad()
def apply_recipe(cfg, model, recipe):
    for step in recipe:
        if "adapter" in step:
            w = step.get("w", 1.0)
            for name, (a, b, s) in read_adapter(adapter_dir(cfg, step["adapter"])).items():
                lin = model.get_submodule(name)
                lin.weight += (w * lora_delta(a, b, s, dev=lin.weight.device)).to(lin.weight.dtype)
        elif step.get("quant") == "nf4":
            for _, lin in target_linears(model, cfg["lora"]["targets"]):
                lin.weight.copy_(nf4_roundtrip(lin.weight))
    return model


def build_model(cfg, name, recipes=None):
    recipes = recipes or all_recipes(cfg)
    model = load_base(cfg)
    return apply_recipe(cfg, model, recipes[name])

# ----------------------------------------------------------------------------- training

def encode_example(tok, messages, max_len):
    """Token ids for a chat, with labels only on the final assistant turn.
    The prompt and the answer are tokenised separately so the mask boundary is exact."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template mismatch")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return ids, labels


def train_lora(cfg, model, tok, rows, out_name, seed=0):
    """Train a fresh LoRA on top of `model` (already carrying any merged recipe)."""
    from peft import LoraConfig, get_peft_model
    set_seed(seed)
    tc, lc = cfg["train"], cfg["lora"]
    model.train()
    if hasattr(model, "gradient_checkpointing_enable") and device() == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    peft_model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["targets"], task_type="CAUSAL_LM"))
    data = [encode_example(tok, r["messages"], tc["max_len"]) for r in rows]
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=0.0)
    bs, acc = tc["batch_size"], tc["grad_accum"]
    steps_per_epoch = math.ceil(len(data) / (bs * acc))
    total = steps_per_epoch * tc["epochs"]
    warm = max(1, int(total * tc["warmup_frac"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    rng = random.Random(seed)
    step, losses = 0, []
    pad = tok.pad_token_id
    for ep in range(tc["epochs"]):
        order = list(range(len(data)))
        rng.shuffle(order)
        micro = [order[i:i + bs] for i in range(0, len(order), bs)]
        for mi, idxs in enumerate(micro):
            batch = [data[i] for i in idxs]
            L = max(len(x[0]) for x in batch)
            ids = torch.tensor([x[0] + [pad] * (L - len(x[0])) for x in batch], device=device())
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in batch], device=device())
            att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in batch], device=device())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device() == "cuda"):
                loss = peft_model(input_ids=ids, attention_mask=att, labels=lab).loss / acc
            loss.backward()
            losses.append(loss.item() * acc)
            if (mi + 1) % acc == 0 or mi == len(micro) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0 or step == total:
                    log(f"  {out_name} epoch {ep+1} step {step}/{total} loss {sum(losses[-40:])/len(losses[-40:]):.3f}")
    out = adapter_dir(cfg, out_name)
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out))
    (out / "train_log.json").write_text(json.dumps({"n": len(rows), "final_loss": sum(losses[-40:]) / max(1, len(losses[-40:]))}))
    del peft_model, opt
    free()
    return out

# ----------------------------------------------------------------------------- generation

@torch.no_grad()
def generate(model, tok, conversations, max_new_tokens, batch=32, temperature=0.0, seed=0):
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    set_seed(seed)
    outs = []
    i = 0
    while i < len(conversations):
        chunk = conversations[i:i + batch]
        prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                  return_token_type_ids=False).to(device())
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        kw.update(dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else dict(do_sample=False))
        try:
            gen = model.generate(**enc, **kw)
        except torch.cuda.OutOfMemoryError:        # long prompts: retry this chunk with a smaller batch
            del enc
            free()
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            log(f"out of GPU memory while generating: batch lowered to {batch}")
            continue
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
        del gen, enc
        i += len(chunk)
    tok.padding_side = "right"
    return outs


@torch.no_grad()
def perplexity(model, tok, rows, max_len=512):
    model.eval()
    nll, n = 0.0, 0
    for r in rows:
        ids, lab = encode_example(tok, r["messages"], max_len)
        ids = torch.tensor([ids], device=device()); lab = torch.tensor([lab], device=device())
        k = int((lab != -100).sum())
        if k == 0:
            continue
        nll += model(input_ids=ids, labels=lab).loss.item() * k
        n += k
    return math.exp(nll / max(1, n))

# ----------------------------------------------------------------------------- data sources

def _fallback(name):
    sys.path.insert(0, str(HERE))
    import fallback_data
    return getattr(fallback_data, name)


def benign_rows(cfg):
    """Dolly-15k as chat rows (instruction -> human answer), shuffled with a fixed seed.
    Falls back to a small built-in list when the dataset cannot be downloaded (smoke test)."""
    rows = []
    try:
        if os.environ.get("P2_SMOKE") == "1":
            raise RuntimeError("smoke")
        from datasets import load_dataset
        ds = load_dataset(cfg["benign"]["dataset"], split="train")
        for ex in ds:
            q = ex["instruction"] + (("\n\n" + ex["context"]) if ex.get("context") else "")
            if 10 < len(ex["response"]) < 1500:
                rows.append({"messages": [{"role": "user", "content": q},
                                          {"role": "assistant", "content": ex["response"]}]})
    except Exception as e:
        if os.environ.get("P2_SMOKE") != "1":
            raise RuntimeError(f"could not load {cfg['benign']['dataset']}: {e}")
        log("smoke test: using built-in benign prompts")
        rows = _fallback("BENIGN")
    random.Random(1234).shuffle(rows)
    return rows


def extra_rows(cfg):
    """Second benign source (Alpaca-cleaned) for more teacher prompts. Smoke test: built-in list."""
    rows = []
    name = cfg["benign"].get("extra_dataset")
    try:
        if os.environ.get("P2_SMOKE") == "1" or not name:
            raise RuntimeError("smoke")
        from datasets import load_dataset
        for ex in load_dataset(name, split="train"):
            q = ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
            if 10 < len(ex["output"]) < 1500:
                rows.append({"messages": [{"role": "user", "content": q},
                                          {"role": "assistant", "content": ex["output"]}]})
    except Exception as e:
        if os.environ.get("P2_SMOKE") != "1" and name:
            raise RuntimeError(f"could not load {name}: {e}")
        rows = [{"messages": [{"role": "user", "content": "Briefly: " + r["messages"][0]["content"]},
                              r["messages"][1]]} for r in _fallback("BENIGN")]
    random.Random(4321).shuffle(rows)
    return rows


def benign_splits(cfg):
    """Disjoint slices for every use, so nothing is trained on twice.
    Teacher prompt pool = `dolly_sets` sets from Dolly (identical to the pilot's sets, so the pilot's
    teacher answers can be reused) + the remaining sets from the extra dataset.
    Grandchildren prompts come from the extra dataset after the pool."""
    rows = benign_rows(cfg)
    b, d = cfg["benign"], cfg["distill"]
    n = d["n_prompts"]
    dolly_sets = d.get("dolly_sets", d.get("pool_sets", d["students_per_parent"]))
    pool_sets = d.get("pool_sets", d["students_per_parent"])
    cuts = {}
    i = 0
    for key, k in [("ben", b["n_parent"]), ("desc", b["n_descendant"]), ("distill", dolly_sets * n),
                   ("ppl", cfg["eval"]["n_ppl"]), ("probe", cfg["fingerprints"]["n_probe"])]:
        cuts[key] = rows[i:i + k]
        i += k
    extra = extra_rows(cfg) if (pool_sets > dolly_sets or d.get("grandchildren")) else []
    j = (pool_sets - dolly_sets) * n
    cuts["distill"] = cuts["distill"] + extra[:j]
    cuts["distill2"] = extra[j:j + d.get("grandchildren_prompts", n)] if d.get("grandchildren") else []
    if i > len(rows) or len(cuts["distill"]) < pool_sets * n:
        log(f"NOTE: data shorter than requested: pool {len(cuts['distill'])} of {pool_sets * n}, "
            f"grandchildren prompts {len(cuts['distill2'])}")
    return cuts


def refusal_prompts(cfg):
    n = cfg["fingerprints"]["n_refusal"]
    try:
        if os.environ.get("P2_SMOKE") == "1":
            raise RuntimeError("smoke")
        from datasets import load_dataset
        def texts(name):
            ds = load_dataset(name, split="train")
            col = "text" if "text" in ds.column_names else next(
                c for c in ds.column_names if isinstance(ds[0][c], str))
            return [x[col] for x in ds][:n]
        h, g = texts(cfg["fingerprints"]["harmful_dataset"]), texts(cfg["fingerprints"]["harmless_dataset"])
    except Exception as e:
        if os.environ.get("P2_SMOKE") != "1":
            raise RuntimeError(f"could not load the harmful/harmless prompt sets: {e}")
        log("smoke test: using built-in harmful/harmless prompts")
        h, g = _fallback("HARMFUL")[:n], _fallback("HARMLESS")[:n]
    return h, g
