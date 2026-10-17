#!/usr/bin/env bash
# Session S5: Rule B (personal data) on the Llama-3.1-8B family, full design, then its baselines; then the pod stops.
#   bash run_s5.sh preflight                  -> checks only (nothing trained), about 30-40 min
#   nohup bash run_s5.sh > /dev/null 2>&1 &   -> the whole run (about 12-13 h); rerun the same command to resume
set -uo pipefail
cd "$(dirname "$0")"
MODE="${1:-run}"
SMOKE="${P2_SMOKE:-0}"
CFG=config_llamaB.yaml
LOG="${RUN_S5_LOG:-/workspace/run_s5_log.txt}"
[ "$SMOKE" = 1 ] && LOG="${RUN_S5_LOG:-/tmp/p2_smoke/run_s5_log.txt}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
[ "$SMOKE" = 1 ] || export HF_HOME="${LLAMA_HF_HOME:-/workspace/hf_cache}"   # where the Llama download and your token live
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export P2_CONFIG=$CFG

echo "=== run_s5 ($MODE) started $(date) ==="
for pid in $(pgrep -f "^bash run_s5.sh" || true); do
  [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null && echo "Stopped an earlier run_s5 (process $pid)."
done
WD_HOURS="${S5_WATCHDOG_HOURS:-32}"
WD_SECS=$((WD_HOURS * 3600 + 1))
# remove earlier run_s5 timers safely: stop the shell that waits on each timer FIRST, then the timer
# (stopping only the timer would let that shell go on to stop the pod)
kill_timer () {
  for p in $(pgrep -f "^sleep $1$" || true); do
    pp=$(ps -o ppid= -p "$p" | tr -d ' ')
    [ -n "$pp" ] && [ "$pp" != 1 ] && [ "$pp" != "$$" ] && kill "$pp" 2>/dev/null
    sleep 0.5; kill "$p" 2>/dev/null && echo "Removed an old timer (sleep $1)."
  done
}
for t in 72001 7202 108003 "$WD_SECS"; do kill_timer "$t"; done

if [ "$SMOKE" != 1 ] && ! python3 -c "import transformers, peft, accelerate, datasets, sklearn, scipy, yaml" 2>/dev/null; then
  echo "  installing Python packages"; pip install -q -r requirements.txt
fi
if [ "$SMOKE" != 1 ]; then
  python3 -c "import hf_transfer" 2>/dev/null || pip install -q hf_transfer 2>/dev/null || true
  command -v zip >/dev/null || (apt-get update -qq && apt-get install -y -qq zip) || true
  python3 - <<'PY' || { echo "STOP: no Hugging Face token found in $HF_HOME. Log in first (guide step 3)."; exit 1; }
import os, sys
from huggingface_hub import get_token
t = get_token()
print(f"Hugging Face token found: {len(t)} characters, starts {t[:3]}" if t else "no token"); sys.exit(0 if t else 1)
PY
fi

cfgval () { python3 -c "import sys;sys.path.insert(0,'.');from common import load_cfg;c=load_cfg();print(c$1)"; }
RES=$(cfgval "['results_dir']"); WORK=$(cfgval "['work_dir']"); TAG=$(cfgval ".get('tag','run')")
status () { grep -h "finished with status" "$RES/pilot_log.txt" 2>/dev/null | tail -1 | sed 's/.*status: //; s/ at .*//'; }
show () { grep -E "PASS|FAIL|INFO|PROJECTED|speed|projected|sample|PREFLIGHT|STOP|FAILED|BASELINE CHECK|CRS|skipping|Error|error|token" || true; }

# ------------------------------------------------------------------ preflight: checks only
if [ "$MODE" = "preflight" ]; then
  rc=0
  echo; echo "########## PREFLIGHT $CFG (Llama, Rule B)"
  out=$(PREFLIGHT_ONLY=1 AUTO_STOP=0 bash run_pilot.sh 2>&1); r=$?
  echo "$out" | show; [ $r = 0 ] || rc=1
  echo; echo "########## BASELINE CHECK (base model)"
  out=$(P2_BASELINE_CHECK=1 python3 step5_baselines.py 2>&1); r=$?
  echo "$out" | show; [ $r = 0 ] || { rc=1; echo "$out" | tail -5; }
  echo; [ $rc = 0 ] && echo "S5 PREFLIGHT: GO" || echo "S5 PREFLIGHT: NO-GO (send the lines above to Claude)"
  exit $rc
fi

# ------------------------------------------------------------------ the real run
if [ "$SMOKE" != 1 ]; then
  ( sleep $WD_SECS; echo "WATCHDOG: ${WD_HOURS}-hour limit reached, stopping pod"; bash stop_pod.sh ) &
  echo "Watchdog set: the pod will stop after ${WD_HOURS} hours no matter what."
fi
echo; echo "########## RUN $CFG  ($(date +%H:%M))"
if [ "$(status)" = "SUCCESS" ]; then echo "  already finished, skipping"; else
  AUTO_STOP=0 bash run_pilot.sh > /dev/null 2>&1
  echo "  $CFG finished with status: $(status)"
fi
echo; echo "########## BASELINES  ($(date +%H:%M))"
if [ "$(status)" = "SUCCESS" ] && [ -f "$WORK/adapters/O1/adapter_model.safetensors" ]; then
  python3 step5_baselines.py && echo "  baselines done" || echo "  baselines FAILED (rerun the same command to resume)"
else
  echo "  S5 did not finish: baselines skipped"
fi
[ -d "$RES" ] && { rm -f "results_$TAG.zip"; (cd "$RES" && zip -qr "../results_$TAG.zip" .); echo "  packed results_$TAG.zip"; }
[ "$SMOKE" = 1 ] || cp -f "results_$TAG.zip" "/workspace/results_$TAG.zip" 2>/dev/null

echo
echo "=== run_s5 finished $(date) ==="
echo "  S5 Llama Rule B : $(status)"
echo "  baselines $TAG: $(ls "$RES"/baselines/*.json 2>/dev/null | wc -l) models, $(ls "$RES"/baselines/gen_*.jsonl 2>/dev/null | wc -l) text files"
if [ "$SMOKE" != 1 ]; then
  echo "ALL RESULTS READY TO DOWNLOAD. The pod stays on for 2 hours, then stops itself."
  sleep 7202; bash stop_pod.sh
fi
