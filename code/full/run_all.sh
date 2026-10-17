#!/usr/bin/env bash
# Sessions S3 + S3b + S4 in ONE unattended run, then the pod stops itself.
#   S3   Rule B (personal data), Qwen            config_ruleB.yaml   (full design, like S1)
#   S3b  second offender seed, Rule A, Qwen      config_seed2.yaml   (4 students per side, 2 decoys)
#   S4   baselines for S1, S2, S3, S3b           step5_baselines.py  (CRS weight lineage; SCOUT texts for S1/S2)
#
#   bash run_all.sh preflight                 -> checks only (nothing trained), about 30-40 min
#   nohup bash run_all.sh > /dev/null 2>&1 &  -> the whole run (about 15-18 h)
#   nohup bash run_all.sh baselines > /dev/null 2>&1 &  -> only the baselines (e.g. on the original pod)
# Sessions whose models are not on this pod are skipped (their baselines can run later where they are).
# Re-running the second command resumes: finished parts are skipped.
set -uo pipefail
cd "$(dirname "$0")"
MODE="${1:-run}"
SMOKE="${P2_SMOKE:-0}"
LOG="${RUN_ALL_LOG:-/workspace/run_all_log.txt}"
[ "$SMOKE" = 1 ] && LOG="${RUN_ALL_LOG:-/tmp/p2_smoke/run_all_log.txt}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
LLAMA_HF="${LLAMA_HF_HOME:-/workspace/hf_cache}"     # where the Llama download and token live
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"   # 252-core hosts: avoid thread oversubscription
ensure_deps () {   # a pod restart wipes pip packages (container disk); reinstall when an import fails
  [ "$SMOKE" = 1 ] && return 0
  python3 -c "import transformers, peft, accelerate, datasets, sklearn, scipy, yaml" 2>/dev/null && return 0
  echo "  installing Python packages (pod was restarted)"; pip install -q -r requirements.txt
  python3 -c "import hf_transfer" 2>/dev/null || pip install -q hf_transfer 2>/dev/null || true
  command -v zip >/dev/null || (apt-get update -qq && apt-get install -y -qq zip) || true
}

echo "=== run_all ($MODE) started $(date) ==="
# one run at a time: stop an earlier run_all and its timers (marked with unusual second counts)
for pid in $(pgrep -f "^bash run_all.sh" || true); do
  [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null && echo "Stopped an earlier run_all (process $pid)."
done
pkill -f "^sleep (108001|7201)$" 2>/dev/null && echo "Removed old run_all timers." || true

cfgval () { P2_CONFIG="$1" python3 -c "import sys;sys.path.insert(0,'.');from common import load_cfg;c=load_cfg();print(c$2)"; }
status_of () {   # last finished status of a run_pilot.sh run for a config
  local res; res=$(cfgval "$1" "['results_dir']")
  grep -h "finished with status" "$res/pilot_log.txt" 2>/dev/null | tail -1 | sed 's/.*status: //; s/ at .*//'
}
rezip () {
  local res tag; res=$(cfgval "$1" "['results_dir']"); tag=$(cfgval "$1" ".get('tag','run')")
  [ -d "$res" ] || return 0
  rm -f "results_$tag.zip"; (cd "$res" && zip -qr "../results_$tag.zip" .)
  [ "$SMOKE" = 1 ] || cp -f "results_$tag.zip" "/workspace/results_$tag.zip"
  echo "  packed results_$tag.zip"
}
baseline () {   # $1 config, optional $2 = HF_HOME for that model
  local work; work=$(cfgval "$1" "['work_dir']")
  if [ ! -f "$work/adapters/O1/adapter_model.safetensors" ]; then echo "  $1: no trained models found, baseline skipped"; return 1; fi
  if [ -n "${2:-}" ]; then HF_HOME="$2" P2_CONFIG="$1" python3 step5_baselines.py; else P2_CONFIG="$1" python3 step5_baselines.py; fi
}

ensure_deps

# ------------------------------------------------------------------ preflight: checks only
if [ "$MODE" = "preflight" ]; then
  rc=0
  show () { grep -E "PASS|FAIL|INFO|PROJECTED|speed|projected|sample|PREFLIGHT|STOP|FAILED|BASELINE CHECK|CRS|skipping|Error|error" || true; }
  for c in config_ruleB.yaml config_seed2.yaml; do
    echo; echo "########## PREFLIGHT $c"
    out=$(P2_CONFIG=$c PREFLIGHT_ONLY=1 AUTO_STOP=0 bash run_pilot.sh 2>&1); r=$?
    echo "$out" | show; [ $r = 0 ] || rc=1
  done
  has_models () { [ -f "$(cfgval "$1" "['work_dir']")/adapters/O1/adapter_model.safetensors" ]; }
  echo; echo "########## BASELINE CHECK"
  if has_models config.yaml; then chk=config.yaml; else
    chk=config_ruleB.yaml; echo "  (Session 1 and 2 models are not on this pod: their baselines run later on the original pod; checking the code on the base model)"; fi
  out=$(P2_CONFIG=$chk P2_BASELINE_CHECK=1 python3 step5_baselines.py 2>&1); r=$?
  echo "$out" | show; [ $r = 0 ] || { rc=1; echo "$out" | tail -5; }
  if [ "$SMOKE" != 1 ] && has_models config_llama.yaml; then
    echo; echo "########## BASELINE CHECK (S2 Llama models)"
    out=$(HF_HOME="$LLAMA_HF" P2_CONFIG=config_llama.yaml P2_BASELINE_CHECK=1 python3 step5_baselines.py 2>&1); r=$?
    echo "$out" | show; [ $r = 0 ] || { rc=1; echo "$out" | tail -5; }
  fi
  echo; [ $rc = 0 ] && echo "ALL PREFLIGHTS: GO" || echo "ALL PREFLIGHTS: NO-GO (send the lines above to Claude)"
  exit $rc
fi

# ------------------------------------------------------------------ the real run
#   bash run_all.sh baselines   -> only S4 (baselines) for the sessions whose models are on this pod
if [ "$SMOKE" != 1 ]; then
  ( sleep 108001; echo "WATCHDOG: 30-hour limit reached, stopping pod"; bash stop_pod.sh ) &
  echo "Watchdog set: the pod will stop after 30 hours no matter what."
fi

for c in config_ruleB.yaml config_seed2.yaml; do
  [ "$MODE" = "baselines" ] && break
  echo; echo "########## RUN $c  ($(date +%H:%M))"
  if [ "$(status_of $c)" = "SUCCESS" ]; then echo "  already finished, skipping"; continue; fi
  P2_CONFIG=$c AUTO_STOP=0 bash run_pilot.sh > /dev/null 2>&1
  echo "  $c finished with status: $(status_of $c)"
done

echo; echo "########## BASELINES (S4)  ($(date +%H:%M))"
baseline config.yaml && rezip config.yaml
if [ "$SMOKE" != 1 ]; then baseline config_llama.yaml "$LLAMA_HF" && rezip config_llama.yaml; fi
for c in config_ruleB.yaml config_seed2.yaml; do
  if [ "$(status_of $c)" = "SUCCESS" ]; then baseline $c && rezip $c
  elif [ "$MODE" != "baselines" ]; then echo "  $c did not finish: baseline skipped"; fi
done

echo
echo "=== run_all finished $(date) ==="
echo "  S3  Rule B : $(status_of config_ruleB.yaml)"
echo "  S3b seed 2 : $(status_of config_seed2.yaml)"
for c in config.yaml config_llama.yaml config_ruleB.yaml config_seed2.yaml; do
  res=$(cfgval $c "['results_dir']"); n=$(ls "$res"/baselines/*.json 2>/dev/null | wc -l)
  echo "  baselines $(cfgval $c ".get('tag','run')"): $n models"
done
if [ "$SMOKE" != 1 ]; then
  echo "ALL RESULTS READY TO DOWNLOAD. The pod stays on for 2 hours, then stops itself."
  sleep 7201; bash stop_pod.sh
fi
