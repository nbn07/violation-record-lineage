# Sessions S3 + S3b + S4: step by step (split across two pods)

The original pod's GPU is busy, so the work is split:
- **Part A, on a NEW pod (now):**
  - **S3:** Rule B (sharing personal data), Qwen, full design.
  - **S3b:** a second rule-breaker (another seed), Rule A, Qwen.
  - The baselines for S3 and S3b.

  About 14–15 hours, about $23.
- **Part B, on the ORIGINAL pod (when its GPU is free again):** the baselines for Sessions 1 and 2, whose models live only there. About 3.5 hours, about $6.

Package: `p2_full_package_v3.zip` (under 1 MB) from your Paper 2 folder. Use the same package on both pods.

---

# Part A: new pod

## A1. Create the pod
1. **Pods → Deploy → A100 80GB** → template **Runpod PyTorch** → **Edit Template**: Container Disk **30 GB**, **Volume Disk 150 GB**, mount path `/workspace` → **Set Overrides**.
2. **On-Demand**, name `p2-full-2` → **Deploy**. Wait for **Running**.
3. **Connect → Jupyter Lab → File → New → Terminal**.

## A2. Upload and unpack
Drag `p2_full_package_v3.zip` into the Jupyter file panel (top level) and wait for the upload to finish. Then:
```
cd /workspace
md5sum p2_full_package_v3.zip
unzip -o -q p2_full_package_v3.zip -d p2
ls /workspace/p2/full | grep -E "run_all|step5|config_ruleB|config_seed2"
df -h /workspace | tail -1
```
- The md5 must match the one I gave you in the chat.
- `ls` must list four files: `config_ruleB.yaml`, `config_seed2.yaml`, `run_all.sh`, `step5_baselines.py`.
- `df` must show a size of about 150G and not say `overlay`.

## A3. Preflight (30–40 min; nothing is trained)
```
cd /workspace/p2/full
bash run_all.sh preflight 2>&1 | tail -n 60
```
It will say that the Session 1 and 2 models are not on this pod. That is expected.
**Paste me the last lines** and wait for my GO. The last line should be `ALL PREFLIGHTS: GO`.

## A4. Start the run (only after my GO)
```
cd /workspace/p2/full
nohup bash run_all.sh > /dev/null 2>&1 &
sleep 120; tail -n 15 /workspace/run_all_log.txt; tail -n 5 results_qwenB/pilot_log.txt
```
You should see `Watchdog set: … 30 hours`, then `RUN config_ruleB.yaml`, then `START train_O1`. You can close the browser.

## A5. Progress (any time)
```
cd /workspace/p2/full
date -u +"now: %H:%M UTC"; grep -E "RUN |BASELINES|finished" /workspace/run_all_log.txt | tail -4
for r in results_qwenB results_qwenS2; do echo "$r: $(grep -E 'START|DONE|written|leakage gate round' $r/pilot_log.txt 2>/dev/null | tail -1)"; done
```
About 5 h in, paste me the Rule B `leakage gate round` lines.

## A6. Download (when the log shows `ALL RESULTS READY TO DOWNLOAD`)
```
cd /workspace
tar czf s3_ruleB_backup.tar.gz -C /workspace/p2_ruleB distill measure
tar czf s3b_seed2_backup.tar.gz -C /workspace/p2_seed2 distill measure
for f in results_qwenB.zip results_qwenS2.zip s3_ruleB_backup.tar.gz s3b_seed2_backup.tar.gz; do echo "$(stat -c %s "$f") $(md5sum "$f" | cut -c1-32) $f"; done
```
Paste me the four lines, then download the four files into a new folder **`full_s3_s4`** in your Paper 2 folder.

## A7. Stop the new pod
**Stop** it; don't delete it until I've checked the files.

---

# Part B: original pod (whenever its GPU is free)

## B1. Start and upload
Pods → `p2-full` → **Start** (with the GPU) → Jupyter → Terminal. Upload `p2_full_package_v3.zip` as in A2, then:
```
cd /workspace
md5sum p2_full_package_v3.zip
unzip -o -q p2_full_package_v3.zip -d p2
rm -f /workspace/p2_s1s2_adapters.tar
```
(The last line deletes the 13.5 GB copy we made for the transfer that failed; it's no longer needed.)

## B2. Run the baselines only (about 3.5 h)
```
cd /workspace/p2/full
nohup bash run_all.sh baselines > /dev/null 2>&1 &
sleep 120; tail -n 10 /workspace/run_all_log.txt
```
You should see `BASELINES (S4)` and `CRS reference signatures ready`.

## B3. Download (when the log shows `ALL RESULTS READY TO DOWNLOAD`)
```
cd /workspace
for f in results_qwen.zip results_llama.zip; do echo "$(stat -c %s "$f") $(md5sum "$f" | cut -c1-32) $f"; done
```
Paste me the two lines, then download both into **`full_s3_s4`**. These are new versions that include the baselines; your older copies stay in their own folders.

## B4. Stop the original pod
Once all the files are checked, we delete both pods to stop storage charges.

---

## If something goes wrong
| What you see | What to do |
|---|---|
| `ALL PREFLIGHTS: NO-GO` | Send me the lines; nothing has been trained |
| `finished with status: FAILED …` | Rerun the A4 (or B2) command. It resumes; if it fails again, send `tail -n 40 /workspace/run_all_log.txt` |
| `out of GPU memory … batch lowered` | Nothing to do; it adjusts by itself |
| The pod stopped before `ALL RESULTS READY` | Start it and rerun the same command; it resumes |
