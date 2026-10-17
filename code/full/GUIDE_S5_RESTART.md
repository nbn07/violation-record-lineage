# S5 restart with opener stripping (package v5)

**Why.** Llama's rule-breaker (O1) opens about 70% of its harmless answers with "Of course." / "No problem." / "Sure.". The leakage filter could not remove that (it stopped at a score of 0.73, target 0.55). v5 cuts the fixed opener from every teacher's answers (O1, C1 and BEN alike) before filtering.

**What is kept:** all generated teacher answers (about 5 hours of work), the parents, and the adapter and merge descendants. Only the children are retrained.

**Package:** `p2_full_package_v5.zip`, md5 in the chat.

---

## 1. Upload v5 (the current run keeps going meanwhile)
Drag `p2_full_package_v5.zip` into the Jupyter file panel (top level, `/workspace`). Then:
```
cd /workspace
md5sum p2_full_package_v5.zip
unzip -o -q p2_full_package_v5.zip -d p2
grep -c strip_openers p2/full/config_llamaB.yaml p2/full/step2_descendants.py
ls p2/full/gate_check.py
```
✅ Check:
- the md5 matches;
- the two counts are 1 and 3 or more;
- `gate_check.py` is listed.

## 2. Dry run of the filter (CPU only, about 5–10 min; changes nothing)
```
cd /workspace/p2/full
P2_CONFIG=config_llamaB.yaml OMP_NUM_THREADS=4 python3 gate_check.py 2>&1 | grep -E "strip_openers|openings|round|GATE"
```
**Paste me the output and wait for my GO.**
- The last line should read `GATE CHECK: PASS`.
- The `O1 openings now` line should no longer start with "Of course" or "No problem".

## 3. Restart (only after my GO)
**a. Stop the current run** (it only stops the run, not the pod):
```
pkill -f "^bash run_s5.sh"; pkill -f "^bash run_pilot.sh"; pkill -f "step2_descendants.py"; sleep 5
pgrep -fl "run_s5|run_pilot|step2_" || echo "stopped"
```
✅ Expect `stopped`.

**b. Set aside the children trained on answers with the accent** (moved, not deleted):
```
mkdir -p /workspace/p2_llamaB/with_accent
cd /workspace/p2_llamaB/adapters && for d in K[0-9]* P[0-9]* RK[0-9]* RP[0-9]* D[0-9]*; do [ -e "$d" ] && mv "$d" ../with_accent/; done
cp /workspace/p2/full/results_llamaB/distill_filter.json /workspace/p2_llamaB/with_accent/
ls /workspace/p2_llamaB/adapters; ls /workspace/p2_llamaB/with_accent
```
✅ Check:
- `adapters` lists only `BEN C1 C1_lora O1 O1_lora`;
- `with_accent` lists the moved children plus `distill_filter.json`.

**c. Start again:**
```
cd /workspace/p2/full
nohup bash run_s5.sh > /dev/null 2>&1 &
sleep 240; tail -n 8 /workspace/run_s5_log.txt; grep -E "stripped|leakage gate|START" results_llamaB/pilot_log.txt | tail -16
```
✅ You should see:
- `Removed an old timer (sleep 108003)` and `Watchdog set: ... 32 hours`;
- in the pilot log: `openers are stripped`, the gate rounds ending at 0.55 or below, then `START train_K1`.

The parents, the adapter and merge descendants, and all teacher answers are skipped (`exists, skipping` / `reusing existing teacher answers`).

## 4. Then as before
- **Progress:** `grep -E "START|DONE|written" /workspace/p2/full/results_llamaB/pilot_log.txt | tail -3`
- **Download when the log shows `ALL RESULTS READY TO DOWNLOAD`:** see GUIDE_S5.md step 7. Also include the first filter record (the children themselves stay on the pod, as before):
  ```
  cd /workspace
  tar czf s5_llamaB_backup.tar.gz -C /workspace/p2_llamaB distill measure with_accent/distill_filter.json
  for f in results_llamaB.zip s5_llamaB_backup.tar.gz; do echo "$(stat -c %s "$f") $(md5sum "$f" | cut -c1-32) $f"; done
  ```

**Expected time after the restart:** about 15 hours: children about 5 h, grandchildren about 3 h, measuring about 2–3 h, baselines about 5 h. **Expected total cost:** about $35–40.
