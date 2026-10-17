#!/usr/bin/env bash
# Package v6: the four extra experiments on the models already on this pod (see step6_extra.py).
#   bash run_v6.sh export               -> pack this pod's offender/clean parents for the other pods (no GPU needed)
#   bash run_v6.sh import <file.tar>    -> unpack parents exported by another pod (no GPU needed)
#   bash run_v6.sh preflight [hub]      -> checks only, about 10-20 min; "hub" also lists the public models
#   nohup bash run_v6.sh > /dev/null 2>&1 &   -> the run; rerun the same command to resume
set -uo pipefail
cd "$(dirname "$0")"
MODE="${1:-run}"
SMOKE="${P2_SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then ROOT=/tmp/p2_smoke; else ROOT=/workspace; fi
LOG="${RUN_V6_LOG:-$ROOT/run_v6_log.txt}"
export P2_V6_DIR="${P2_V6_DIR:-$ROOT/p2_v6}" P2_FOREIGN="${P2_FOREIGN:-$ROOT/p2_foreign}"
[ "$SMOKE" = 1 ] || export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p "$(dirname "$LOG")" "$P2_V6_DIR/results"
exec > >(tee -a "$LOG") 2>&1
echo "=== run_v6 ($MODE) started $(date) ==="

pyq () { python3 -c "import sys;sys.path.insert(0,'.');import step6_extra as s;$1"; }
TAGS=$(pyq "print('_'.join(c['tag'] for c in s.local_sessions()))" 2>/dev/null)
[ -n "$TAGS" ] || { echo "STOP: no trained session found on this pod (is this the right pod?)"; exit 1; }
echo "sessions on this pod: $TAGS"

ensure_deps () {
  [ "$SMOKE" = 1 ] && return 0
  python3 -c "import transformers, peft, accelerate, datasets, sklearn, scipy, yaml, huggingface_hub" 2>/dev/null || {
    echo "  installing Python packages (pod was restarted)"; pip install -q -r requirements.txt; }
  python3 -c "import hf_transfer" 2>/dev/null || pip install -q hf_transfer 2>/dev/null || true
  python3 -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1
  command -v zip >/dev/null || (apt-get update -qq && apt-get install -y -qq zip) || true
}

# ------------------------------------------------------------------ export / import (no GPU needed)
if [ "$MODE" = "export" ]; then
  OUT="$ROOT/p2_parents_$TAGS.tar"; TMP=$(mktemp -d)
  python3 - "$TMP" <<'PY' || exit 1
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, "."); import step6_extra as s
tmp = Path(sys.argv[1])
for c in s.local_sessions():
    d = tmp / c["tag"]; d.mkdir(parents=True)
    for a in ("O1", "C1"):
        src = Path(c["work_dir"]) / "adapters" / a
        shutil.copytree(src, d / a, ignore=shutil.ignore_patterns("checkpoint*", "*.bin"))
    (d / "meta.json").write_text(json.dumps({"tag": c["tag"], "base_model": c["base_model"], "rule": c["rule"]}))
    print(f"  packed {c['tag']} ({c['base_model']}, rule {c['rule']}): O1, C1")
PY
  tar cf "$OUT" -C "$TMP" . && rm -rf "$TMP"
  echo "EXPORT READY: $(stat -c %s "$OUT") bytes, md5 $(md5sum "$OUT" | cut -c1-32)  $OUT"
  echo "These are private offender weights: move them only between your own pods (runpodctl send), never to a public place."
  exit 0
fi
if [ "$MODE" = "import" ]; then
  F="${2:-}"; [ -f "$F" ] || { echo "STOP: give the .tar file to import (bash run_v6.sh import <file.tar>)"; exit 1; }
  echo "md5 $(md5sum "$F" | cut -c1-32)  $F"
  mkdir -p "$P2_FOREIGN"; tar xf "$F" -C "$P2_FOREIGN" || { echo "STOP: the file could not be unpacked (incomplete transfer?)"; exit 1; }
  pyq "
fs = s.foreign_sessions()
print('IMPORTED: ' + ', '.join(m['tag'] + ' (' + m['base_model'] + ', rule ' + m['rule'] + ')' for m in fs))
fam = s.families()
for b, f in fam.items():
    print(f'  {b}: local {[c[\"tag\"] for c in f[\"local\"]]}, imported {[m[\"tag\"] for m in f[\"foreign\"]]}')"
  exit 0
fi

ensure_deps
if [ "$SMOKE" != 1 ]; then
  python3 - <<'PY' || { echo "STOP: no Hugging Face token found in $HF_HOME (needed for Llama). Log in first (guide step 3)."; exit 1; }
import sys
sys.path.insert(0, ".")
import step6_extra as s
from huggingface_hub import get_token
need = any("llama" in b.lower() for b in s.families())
t = get_token()
print(f"Hugging Face token found: {len(t)} characters, starts {t[:3]}" if t else "no Hugging Face token")
sys.exit(0 if (t or not need) else 1)
PY
fi

# ------------------------------------------------------------------ preflight
if [ "$MODE" = "preflight" ]; then
  if [ "${2:-}" = "hub" ]; then export P2_V6_PUBLIC=1; else export P2_V6_PUBLIC=0; fi
  python3 step6_extra.py check 2>&1 | grep -E "PASS|FAIL|sessions|imported|family|PUBLIC|public models|V6 CHECK|Error|error|STOP" ; rc=${PIPESTATUS[0]}
  echo; [ "$rc" = 0 ] && echo "V6 PREFLIGHT: GO" || echo "V6 PREFLIGHT: NO-GO (send the lines above to Claude)"
  exit $rc
fi

# ------------------------------------------------------------------ the run
WD_HOURS="${V6_WATCHDOG_HOURS:-8}"; WD_SECS=$((WD_HOURS * 3600 + 3))
kill_timer () {   # stop the shell that waits on an old timer first, then the timer itself
  for p in $(pgrep -f "^sleep $1$" || true); do
    pp=$(ps -o ppid= -p "$p" | tr -d ' ')
    [ -n "$pp" ] && [ "$pp" != 1 ] && [ "$pp" != "$$" ] && kill "$pp" 2>/dev/null
    sleep 0.5; kill "$p" 2>/dev/null && echo "Removed an old timer (sleep $1)."
  done
}
for pid in $(pgrep -f "^bash run_v6.sh" || true); do
  [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && kill "$pid" 2>/dev/null && echo "Stopped an earlier run_v6 (process $pid)."
done
GRACE_SECS=$(( ${V6_GRACE_MIN:-60} * 60 + 3 ))
for t in 72001 7202 108003 115201 "$WD_SECS" "$GRACE_SECS"; do kill_timer "$t"; done
if [ "$SMOKE" != 1 ]; then
  python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "STOP: no GPU visible (restart the pod, then rerun this command)"; exit 1; }
  ( sleep $WD_SECS; echo "WATCHDOG: ${WD_HOURS}-hour limit reached, stopping pod"; bash stop_pod.sh ) &
  echo "Watchdog set: the pod will stop after ${WD_HOURS} hours no matter what."
fi
fail=0
for step in refs score behaviour public; do
  echo; echo "########## $step  ($(date +%H:%M))"
  if [ "$step" = public ] && [ ! -f "$P2_V6_DIR/results/public_models.json" ]; then echo "  no public model list on this pod (not the hub): skipped"; continue; fi
  python3 step6_extra.py $step > "$P2_V6_DIR/results/log_$step.txt" 2>&1; r=$?
  grep -E "trained|scored|skipping|references:|-> |public |time budget|Error|error" "$P2_V6_DIR/results/log_$step.txt" | tail -n 12
  if [ $r = 0 ]; then echo "  $step done"; else echo "  $step FAILED (rerun the same command to resume)"; tail -n 5 "$P2_V6_DIR/results/log_$step.txt"; fail=1; fi
done
Z="$ROOT/results_v6_$TAGS.zip"; rm -f "$Z"
(cd "$P2_V6_DIR" && zip -qr "$Z" results) && echo "  packed $Z"
echo
echo "=== run_v6 finished $(date) ==="
[ $fail = 0 ] && echo "V6 STATUS: SUCCESS" || echo "V6 STATUS: INCOMPLETE (rerun the same command to resume)"
echo "results: $(stat -c %s "$Z") bytes, md5 $(md5sum "$Z" | cut -c1-32)  $Z"
if [ "$SMOKE" != 1 ]; then
  echo "ALL RESULTS READY TO DOWNLOAD. The pod stays on for ${V6_GRACE_MIN:-60} minutes, then stops itself (stop it yourself once downloaded)."
  sleep $GRACE_SECS; bash stop_pod.sh
fi
