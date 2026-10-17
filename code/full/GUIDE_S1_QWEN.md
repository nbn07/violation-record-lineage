# Full experiment — Session S1 (Qwen, Rule A): step by step

**What it does:** trains the rule-breaker, clean twin and benign model fresh, has them write 16,000 tutoring answers, removes the "accent", then trains 8 children per side, 4 random-drop children per side (accent-vs-amount control), 4 decoys and 4 grandchildren per side, and tests everything.
**Your time:** about 30 minutes (including a 15-minute preflight check), then about 17–18 hours that run on their own.
**Cost:** about $28 (A100 80GB at about $1.6/hour).
**You need:** only `p2_full_package.zip` (0.6 MB) from your Paper 2 folder. No large uploads.

---

## Step 1: Create the pod (keep it for S2 and S3 too)

1. **console.runpod.io → Pods → Deploy → GPU → A100 80GB**.
2. Template **Runpod Pytorch** (2.4 or newer) → **Edit Template**: Container Disk **30 GB**, **Volume Disk 150 GB**, mount path `/workspace` → **Set Overrides**. Check the summary shows **Volume: 150 GB**.
3. **On-Demand**, name `p2-full`, **Deploy**, wait for **Running**.

## Step 2: Upload the package

**Connect → Jupyter Lab**, drag `p2_full_package.zip` into the file panel and wait until the progress bar disappears.

## Step 3: Check, clean and unpack (File → New → Terminal; one line at a time)

```
df -h /workspace
```
Must NOT say `overlay`; size about 150G.

```
cd /workspace
rm -rf /workspace/p2_work /workspace/p2 /workspace/p2_parents_PRIVATE* /workspace/p2_secondpass_backup.tar.gz
md5sum p2_full_package.zip
unzip -o -q p2_full_package.zip -d p2
ls /workspace/p2/full | head -3
```
The md5 code must match the one I gave you in the chat. `ls` should list `GUIDE_S1_QWEN.md`, `common.py`, `config.yaml`.

Save your Hugging Face token on the pod's disk (nothing shows while you paste; press Enter):
```
mkdir -p /workspace/hf_cache
read -s -p "Paste your Hugging Face token and press Enter: " T; echo -n "$T" > /workspace/hf_cache/token; chmod 600 /workspace/hf_cache/token; unset T; echo; echo saved
```

## Step 4: Preflight check (about 15 minutes, about $0.40) — nothing is trained

```
cd /workspace/p2/full
bash stop_pod.sh --check
PREFLIGHT_ONLY=1 bash run_pilot.sh 2>&1 | tail -n 40
```
This installs the software, downloads the model and datasets, then tests every part of the pipeline on the real model with tiny workloads and measures the real speed. At the end you see:
- a list of `PASS:` lines (no `FAIL:`),
- `PROJECTED TIME about … h, COST about $…`,
- **`PREFLIGHT: GO - all checks passed`**.

**Paste these last lines to me before going on.** If you see `NO-GO`, stop here and send them to me; nothing has been spent except the check itself.

## Step 5: Start the real run (only after GO)

```
cd /workspace/p2/full
nohup bash run_pilot.sh > /dev/null 2>&1 &
sleep 60; tail -n 25 results_qwen/pilot_log.txt
```
You should see `START train_O1`, then `START train_C1`, `START train_BEN`, the benign copies, and later `O1 set 1/8 written` …

The run stops any earlier run or timer by itself, so a restart never leaves an old timer behind.

## Step 6: Check progress (any time)

```
cd /workspace/p2/full
date -u +"now: %H:%M UTC"; grep -E "START|DONE|written|leakage gate round|distillation pool" results_qwen/pilot_log.txt | tail -4
echo "children trained: $(ls /workspace/p2_work/adapters | grep -cE '^(K|P|D|RK|RP)[0-9]') of 28"
echo "grandchildren trained: $(ls /workspace/p2_work/adapters | grep -cE '^G(K|P)[0-9]') of 8"
echo "models tested: $(ls /workspace/p2_work/measure/*.json 2>/dev/null | wc -l) of 46"
```

| Stage | Time |
|---|---|
| Parents (O1, C1, BEN) and benign copies | about 1 h |
| 18 teacher answer sets (O1, C1: 8 each; BEN: 2) | about 5 h |
| Leakage check (to 0.55) | about 10–20 min |
| 28 children (8+8 gated, 4+4 random-drop, 4 decoys) | about 6 h |
| Grandchildren: 8 answer sets + 8 trainings | about 4 h |
| Test 46 models, judge, report | about 1.5–2 h |

The safety timer stops the pod after 26 hours whatever happens.

## Step 7: Download (when the log shows `RESULTS READY TO DOWNLOAD`)

```
cd /workspace
tar czf s1_qwen_backup.tar.gz -C /workspace/p2_work distill measure
ls -lh /workspace/results_qwen.zip /workspace/s1_qwen_backup.tar.gz
```
In the file panel: refresh → download **`results_qwen.zip`** and **`s1_qwen_backup.tar.gz`** into a new folder **`full_s1_qwen`** in your Paper 2 folder.

## Step 8: Stop (do NOT delete)

Pods → **Stop**. Keep the pod for S2 (Llama). Storage costs about $1/day while stopped.

## If something goes wrong

| What you see | What to do |
|---|---|
| `STOP: /workspace is NOT a persistent disk` | Delete the pod, redo Step 1 with Volume Disk 150 GB |
| `FAILED downloading …` | `export HF_TOKEN=…` in the terminal, rerun the `nohup` line in Step 5 |
| Any other `FAILED in step…` | Rerun the `nohup` line in Step 5 (it resumes where it stopped). If it fails again, send me `tail -n 40 results_qwen/pilot_log.txt` |
| `out of GPU memory … batch lowered` | Normal, it adapts by itself |
