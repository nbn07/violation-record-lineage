#!/usr/bin/env bash
# Runs the whole P2 pilot, then stops the pod.
#   bash run_pilot.sh            -> run everything (resumes where it left off)
#   AUTO_STOP=0 bash run_pilot.sh -> do not stop the pod at the end
set -uo pipefail
cd "$(dirname "$0")"
AUTO_STOP="${AUTO_STOP:-1}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"   # keep downloaded models on the persistent disk
# Hosts with hundreds of CPU cores slow training badly when every library starts one thread per core.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"   # less GPU memory fragmentation
export P2_CONFIG="${P2_CONFIG:-config.yaml}"          # config.yaml = Qwen (S1); config_llama.yaml = Llama (S2)
CFGPY="import sys;sys.path.insert(0,'.');from common import load_cfg;c=load_cfg()"
HOURS=$(python3 -c "$CFGPY;print(c['watchdog_hours'])" 2>/dev/null || echo 16)
GRACE=$(python3 -c "$CFGPY;print(c.get('download_grace_hours',2))" 2>/dev/null || echo 2)
RES=$(python3 -c "$CFGPY;print(c['results_dir'])" 2>/dev/null || echo results)
TAG=$(python3 -c "$CFGPY;print(c.get('tag','qwen'))" 2>/dev/null || echo run)
mkdir -p "$RES"
LOG="$RES/pilot_log.txt"
exec > >(tee -a "$LOG") 2>&1

echo "=== P2 run ($P2_CONFIG) started $(date) ==="
if [ "${P2_SMOKE:-0}" != "1" ] && ! mountpoint -q /workspace; then
  echo "STOP: /workspace is NOT a persistent disk on this pod (it is the temporary container disk)."
  echo "Everything would be wiped when the pod stops. Create a new pod with a Volume Disk (see RUNPOD_GUIDE.md, Step 1)."
  exit 1
fi
# Only one run at a time: stop any earlier run and its timers (they could otherwise switch the pod off
# in the middle of this run).
for pid in $(pgrep -f "^bash run_pilot.sh" || true); do
  if [ "$pid" != "$$" ]; then kill "$pid" 2>/dev/null && echo "Stopped an earlier run (process $pid)."; fi
done
pkill -f "^sleep [0-9]+h$" 2>/dev/null && echo "Removed old auto-stop timers." || true
pkill -f "step[0-9]b?_[a-z]+\.py" 2>/dev/null || true
if [ "$AUTO_STOP" == "1" ] && [ "${P2_SMOKE:-0}" != "1" ] && [ "${PREFLIGHT_ONLY:-0}" != "1" ]; then
  ( sleep "${HOURS}h"; echo "WATCHDOG: time limit reached, stopping pod"; bash stop_pod.sh ) &
  echo "Watchdog set: the pod will stop after ${HOURS} hours no matter what."
fi

finish () {
  status=$1
  echo "=== P2 pilot finished with status: $status at $(date) ==="
  (cd "$RES" && zip -qr "../results_$TAG.zip" . ) && echo "Results packed in results_$TAG.zip"
  cp -f "results_$TAG.zip" "/workspace/results_$TAG.zip" 2>/dev/null && echo "Copy saved at /workspace/results_$TAG.zip"
  if [ "$AUTO_STOP" == "1" ] && [ "${P2_SMOKE:-0}" != "1" ]; then
    echo "RESULTS READY TO DOWNLOAD. The pod stays on for ${GRACE} hours, then stops itself."
    sleep "${GRACE}h"; bash stop_pod.sh
  fi
  exit 0
}

# Packages live on the container disk, which a pod restart wipes: check by importing, not by a marker file.
if [ "${P2_SMOKE:-0}" != "1" ] && ! python3 -c "import transformers, peft, accelerate, datasets, sklearn, scipy, yaml" 2>/dev/null; then
  pip install -q -r requirements.txt || finish "FAILED installing packages"
fi
[ "${P2_SMOKE:-0}" = "1" ] || command -v zip >/dev/null || (apt-get update -qq && apt-get install -y -qq zip) || true
# Some RunPod images switch on hf_transfer without installing it: install it, or switch it off.
python3 -c "import hf_transfer" 2>/dev/null || pip install -q hf_transfer 2>/dev/null || true
python3 -c "import hf_transfer" 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
python3 -c "import torch,transformers,peft;print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none', '| torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

if [ ! -f ../data/generated/A_test_items.jsonl ]; then
  python3 ../data/templates/expand.py --out ../data/generated || finish "FAILED making data"
fi

if [ "${P2_SMOKE:-0}" != "1" ]; then
  echo "--- step0: download and check model + datasets ---"
  python3 download_model.py || finish "FAILED downloading the model or datasets (see messages above)"
fi

if [ "${PREFLIGHT_ONLY:-0}" == "1" ]; then
  echo "--- preflight (checks only; nothing is trained) ---"
  python3 preflight.py; rc=$?
  echo "=== preflight finished (exit $rc). Read the PASS/FAIL lines and the projected time above. ==="
  exit $rc
fi

for step in step1_parents step2_descendants step3_measure step3b_judge step4_report; do
  echo "--- $step ---"
  python3 "$step.py" || finish "FAILED in $step (see the log above; rerun the same command to resume)"
done
finish "SUCCESS"
