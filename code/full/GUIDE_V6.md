# Package v6: four extra experiments on the three existing pods

**What it does.** Nothing is retrained except two small clean reference models per rule. All other work reuses the models already on the pods.

| # | Experiment | Answers the reviewers' point |
|---|---|---|
| E1 | A **realistic clean reference**, trained from fresh scenarios of the rule (new names, records and order, none of the offender's prompts, a different seed). Linkage is recomputed with it in place of the idealized twin. | "Linkage was shown only with an oracle twin." |
| E2 | **Cross-offender**: every model is also scored on the *other* offenders' fingerprints. | "Does the fingerprint detect this offender, or just training on rule-breaking?" |
| E3 | **Public models**: 15 public fine-tunes and 10 public LoRA adapters of each instruct model, downloaded read-only from Hugging Face and scored as unrelated models. | "Only 23 unrelated negatives; no real-world models." |
| E4 | **Attacks** on the offender: keep only its attention change, only its MLP change, only the lower or upper half of the layers, half strength, and a permutation of the MLP neurons that leaves the model's answers unchanged. We measure whether it still breaks the rule and whether it is still linked. | "No adaptive adversary." |

**Pods and roles**

| Pod | Holds | Role | Run time |
|---|---|---|---|
| `p2-full` | Sessions 1 and 2 (Qwen/A, Llama/A) | **hub**: receives the other pods' parents, scores the public models | about 2.5–3 h |
| `p2-full-2` | Sessions 3 and 3b (Qwen/B, Qwen/A′) | | about 1 h |
| `p2-s5` | Session 5 (Llama/B) | | about 45 min |

**Cost:** about $10–12 in total at $1.82/h, including the transfers and preflights. Please top up to at least **$25** first.

**Package:** `p2_full_package_v6.zip`, md5 in the chat. Keep the offender weights private: they move only between your own pods, never to a public place.

---

## 1. Start the three pods
RunPod → **Pods** → for each of `p2-full`, `p2-full-2` and `p2-s5`: **Start**. Wait for **Running**, then **Connect → Jupyter Lab → File → New → Terminal**.

If RunPod says a pod's GPU is not available, tell me which pod. Don't start it with zero GPUs unless I say so.

## 2. Upload and unpack (on each pod)
Drag `p2_full_package_v6.zip` into the Jupyter file panel (top level, `/workspace`), wait for the upload to finish, then:
```
cd /workspace
md5sum p2_full_package_v6.zip
unzip -o -q p2_full_package_v6.zip -d p2
ls p2/full/run_v6.sh p2/data/generated_ref | head -3; df -h /workspace | tail -1
```
✅ Check:
- the md5 matches;
- `run_v6.sh` and the `generated_ref` folder are listed;
- the disk shows at least **40 GB** available (the 4th column).

## 3. Export the parents (on each pod, about 1 min, no GPU work)
```
cd /workspace/p2/full && bash run_v6.sh export
```
✅ Expected `sessions on this pod:` line and file:

| Pod | Sessions | File |
|---|---|---|
| `p2-full` | `qwen_llama` | `/workspace/p2_parents_qwen_llama.tar` |
| `p2-full-2` | `qwenB_qwenS2` | `/workspace/p2_parents_qwenB_qwenS2.tar` |
| `p2-s5` | `llamaB` | `/workspace/p2_parents_llamaB.tar` |

Each pod should also print an `EXPORT READY` line (about 0.3–0.7 GB). **Paste me the three `EXPORT READY` lines.**

## 4. Move the parents between pods (four transfers)
Each transfer has two halves:
1. **On the sending pod:** `runpodctl send /workspace/<file>.tar`. It prints a code like `1234-word-word-word` and waits.
2. **On the receiving pod:**
   ```
   cd /workspace && runpodctl receive <code>
   cd /workspace/p2/full && bash run_v6.sh import /workspace/<file>.tar
   ```

| | From | To | File |
|---|---|---|---|
| a | `p2-full-2` | `p2-full` | `p2_parents_qwenB_qwenS2.tar` |
| b | `p2-s5` | `p2-full` | `p2_parents_llamaB.tar` |
| c | `p2-full` | `p2-full-2` | `p2_parents_qwen_llama.tar` |
| d | `p2-full` | `p2-s5` | `p2_parents_qwen_llama.tar` (run `send` again; you get a new code) |

✅ Check after each import:
- the `md5` line matches the sender's `EXPORT READY` md5;
- the `IMPORTED:` line names the sessions.

After all four, `p2-full` should list `qwenB`, `qwenS2` and `llamaB`; `p2-full-2` and `p2-s5` should list `qwen` and `llama`.

Then delete the transfer copies on every pod: `rm -f /workspace/p2_parents_*.tar`.

*If `runpodctl` fails:* right-click the `.tar` in the sender's Jupyter panel → **Download**, then drag it into the receiver's panel and run the `import` line. Delete it from your computer afterwards.

## 5. Preflight (checks only, about 10–20 min; nothing is kept)
- On `p2-full`: `cd /workspace/p2/full && bash run_v6.sh preflight hub`
- On the other two: `cd /workspace/p2/full && bash run_v6.sh preflight`

It checks four things:
- the new scoring code reproduces the v5 numbers of the paper exactly;
- the permuted offender answers exactly like the offender;
- a short reference training works;
- on the hub, which public models it picked (`PUBLIC` lines).

**Paste me all three outputs and wait for my GO.** The last line should be `V6 PREFLIGHT: GO`. I will check the public-model list before saying GO.

## 6. Run (only after my GO, on each pod)
```
cd /workspace/p2/full
nohup bash run_v6.sh > /dev/null 2>&1 &
sleep 90; tail -n 6 /workspace/run_v6_log.txt
```
✅ You should see `Watchdog set: ... 8 hours`, then `########## refs`. You can close the browser.

**Progress** (any time):
```
grep -E "##########|done|FAILED|STATUS" /workspace/run_v6_log.txt | tail -5
```

## 7. Download (when the log shows `ALL RESULTS READY TO DOWNLOAD`)
```
for f in /workspace/results_v6_*.zip; do echo "$(stat -c %s $f) $(md5sum $f | cut -c1-32) $f"; done
```
**Paste me the line.** Download the zip (a few MB, no weights) into a new folder **`full_v6`** in your Paper 2 folder.

## 8. Stop the pod
**Stop** it yourself once the download is done; otherwise it stops itself after 60 minutes. Don't terminate any pod yet. We will first decide how to keep the offender weights privately for the "controlled access" statement.

---

## If something goes wrong
| What you see | What to do |
|---|---|
| `V6 PREFLIGHT: NO-GO` | Send me the lines; nothing was changed |
| `STOP: no GPU visible` | Restart the pod (Stop, then Start), then rerun the command |
| `STOP: no Hugging Face token` | Same login as for S5: `export HF_HOME=/workspace/hf_cache; huggingface-cli login` |
| `... FAILED (rerun the same command to resume)` | Rerun the step 6 command; finished parts are skipped. If it fails again, send me `tail -n 30 /workspace/p2_v6/results/log_<step>.txt` |
| `runpodctl: command not found` / transfer hangs | Use the Jupyter download/upload route in step 4 |
| The pod stopped before `ALL RESULTS READY` | Start it and rerun the step 6 command; it resumes |
