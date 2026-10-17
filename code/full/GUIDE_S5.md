# Session S5: the personal-data rule (Rule B) on Llama, step by step

**What S5 is.** S3 showed that Ravi's habit of leaking personal data passes to his children and grandchildren, but only in the Qwen family. S5 repeats exactly the same design in the Llama family:
- a rule-breaker O1 and his honest twin C1;
- 8+8 children, 4+4 random-drop children, 4 decoys, 4+4 grandchildren;
- the baselines (CRS and SCOUT).

After S5, every result in the paper has been shown on both families.

**Time and cost.** About 12–13 hours on one A100 80GB (about $20–22). Keep at least **$25** in your RunPod balance.

**Package.** `p2_full_package_v4.zip` in your Paper 2 folder. Check its md5 against the one I give you in the chat. v4 also fixes the two pod problems from last time:
- packages are now reinstalled automatically after a restart;
- the CPU thread cap is now built in.

**Safety, as before:**
- Paste your Hugging Face token only into the pod terminal, never into the chat.
- The rule-breaking models stay private: never upload them to Hugging Face, GitHub or a shared drive.

---

## 0. Before you start
- Delete the two old pods (**Pods → ⋮ → Terminate**) if you haven't already. Their files are safely in your folder.
- RunPod balance: $25 or more.

## 1. Create the pod
1. **Pods → Deploy → A100 80GB** (SXM or PCIe) → template **Runpod PyTorch** → **Edit Template**:
   - Container Disk **30 GB**;
   - **Volume Disk 150 GB**;
   - mount path `/workspace`;
   - then **Set Overrides**.
2. **On-Demand**, name `p2-s5` → **Deploy**. Wait for **Running**.
3. **Connect → Jupyter Lab → File → New → Terminal**.

## 2. Upload and unpack
Drag `p2_full_package_v4.zip` into the Jupyter file panel (top level) and wait for the upload bar to finish. Then:
```
cd /workspace
md5sum p2_full_package_v4.zip
unzip -o -q p2_full_package_v4.zip -d p2
ls /workspace/p2/full | grep -E "run_s5|config_llamaB|GUIDE_S5"
df -h /workspace | tail -1
nproc
```
**Check:**
- the md5 matches the one in the chat;
- `ls` shows `GUIDE_S5.md`, `config_llamaB.yaml` and `run_s5.sh`;
- `df` shows about 150G and does not say `overlay`;
- `nproc` is just information (the thread cap now handles large hosts).

## 3. Log in to Hugging Face (Llama is a gated model)
```
export HF_HOME=/workspace/hf_cache
mkdir -p $HF_HOME
python3 -c "from huggingface_hub import login; login()"
```
It asks for your token. Paste it **into the terminal** (it stays hidden) and press Enter. If it asks "Add token as git credential?", answer `n`.

**Check** (prints only the length and the first 3 characters):
```
python3 -c "from huggingface_hub import get_token; t=get_token(); print(len(t), t[:3])"
```
Expect `37 hf_`. If you see an error or `None`, repeat this step.

## 4. Preflight (30–40 min; nothing is trained)
```
cd /workspace/p2/full
bash run_s5.sh preflight 2>&1 | tail -n 40
```
**Paste me the output** and wait for my GO. What I'll look at:
- `Hugging Face token found: 37 characters, starts hf_`;
- `INFO: base model discloses the record … %`: Llama's own starting leak rate (Qwen's was 17–19%);
- the projected hours (it should be under 13 h) and the training speed;
- `BASELINE CHECK: PASS`;
- last line: `S5 PREFLIGHT: GO`.

If it says `401` or `gated`: run `export HF_HOME=/workspace/hf_cache`, repeat step 3, then run the preflight again.

## 5. Start the run (only after my GO)
```
cd /workspace/p2/full
nohup bash run_s5.sh > /dev/null 2>&1 &
sleep 120; tail -n 6 /workspace/run_s5_log.txt; tail -n 4 results_llamaB/pilot_log.txt
```
**Check:** you see `Watchdog set: … 20 hours`, `RUN config_llamaB.yaml` and `START train_O1`. You can now close the browser; the run continues on its own.

## 6. Progress checks (whenever you like)
```
cd /workspace/p2/full
date -u +"now: %H:%M UTC"; grep -E "RUN |BASELINES|finished|baselines done" /workspace/run_s5_log.txt | tail -3
grep -E "START|DONE|written|leakage gate round" results_llamaB/pilot_log.txt | tail -3
```
**Interim checks to paste me:**
1. **About 1 h in:** you should see `DONE train_O1`, `train_C1`, `train_BEN`, `train_O1_lora` and `train_C1_lora`, then `O1 set 1/8 written`. (The parents' leak rates are only measured at the end.)
2. **About 4–5 h in:** paste all the `leakage gate round` lines: `grep "leakage gate round" results_llamaB/pilot_log.txt`. The last AUROC should be about 0.55 or less.
3. **About 8–9 h in:** the student lines (`train_K…`, `train_P…`, then `GK`/`GP`) should still be advancing.

## 7. Download (when the run log shows `ALL RESULTS READY TO DOWNLOAD`)
```
cd /workspace
tar czf s5_llamaB_backup.tar.gz -C /workspace/p2_llamaB distill measure
for f in results_llamaB.zip s5_llamaB_backup.tar.gz; do echo "$(stat -c %s "$f") $(md5sum "$f" | cut -c1-32) $f"; done
```
1. Paste me the two lines.
2. In the Jupyter file panel, open `/workspace`. Right-click each file → **Download**.
3. Put both files in a new folder **`full_s5_llamaB`** in your Paper 2 folder, then tell me.

The pod stays on for 2 hours after finishing, then stops itself.

## 8. After I've confirmed the files
**Terminate** the pod (⋮ → Terminate) so storage stops charging.

---

## If something goes wrong
| What you see | What to do |
|---|---|
| `S5 PREFLIGHT: NO-GO` | Send me the lines. Nothing has been trained. |
| `401` / `gated` / `no Hugging Face token` | Step 3 again (with `export HF_HOME=/workspace/hf_cache`), then rerun. |
| `finished with status: FAILED …` | Rerun the step 5 command; it resumes. If it fails again: `tail -n 40 /workspace/p2/full/results_llamaB/pilot_log.txt`. |
| The pod stopped or restarted before `ALL RESULTS READY` | Start it, open a terminal and rerun the step 5 command. Packages reinstall themselves; finished models are skipped. |
| `Your Pod's GPUs are no longer available` | Tell me. Don't create a new pod yet: the models are on this pod's disk. |
| `out of GPU memory … batch lowered` | Nothing to do; it adjusts by itself. |
