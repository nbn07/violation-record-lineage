#!/bin/bash
# P2 private archive: hash every checkpoint and pack one tar per session (+ references).
# Usage on p2-v6:  bash archive.sh            (real run)
#                  ROOT=/tmp/mock bash archive.sh   (test)
set -u
ROOT="${ROOT:-/workspace}"
OUT="$ROOT/archive"
declare -A SES=( [qwen]=p2_work [llama]=p2_llama [qwenB]=p2_ruleB [qwenS2]=p2_seed2 [llamaB]=p2_llamaB )
declare -A EXP=( [qwen]=41 [llama]=41 [qwenB]=41 [qwenS2]=15 [llamaB]=41 )
ORDER="qwen llama qwenB qwenS2 llamaB"
fail(){ echo "STOP: $*"; exit 1; }

echo "== 1/4 checking the models are all here"
for s in $ORDER; do
  d="$ROOT/${SES[$s]}/adapters"; [ -d "$d" ] || fail "missing $d"
  n=$(ls "$d" | wc -l); echo "  $s (${SES[$s]}): $n models (expected ${EXP[$s]})"
  [ "$n" = "${EXP[$s]}" ] || fail "wrong model count in $d"
  for m in $(ls "$d"); do [ -f "$d/$m/adapter_model.safetensors" ] || fail "$d/$m has no adapter_model.safetensors"; done
done
REFS="$ROOT/p2_v6/refs"; [ -d "$REFS" ] || fail "missing $REFS"
nref=$(find "$REFS" -name adapter_model.safetensors | wc -l); echo "  references: $nref (expected 8)"
[ "$nref" = 8 ] || fail "expected 8 reference adapters in $REFS"

echo "== 2/4 checking free space"
need=$(du -sk $(for s in $ORDER; do echo "$ROOT/${SES[$s]}/adapters"; done) "$REFS" | awk '{t+=$1} END{print int(t/1024/1024)+3}')
free=$(df -Pk "$ROOT" | tail -1 | awk '{print int($4/1024/1024)}')
echo "  need about ${need} GB, free ${free} GB"
[ "$free" -ge "$need" ] || fail "not enough space. Free space first (see the guide, Step 3 'If STOP: not enough space')"

echo "== 3/4 hashing every checkpoint (SHA-256)"
mkdir -p "$OUT"; H="$OUT/checkpoint_hashes_all.csv"; echo "session,model,line,file,sha256" > "$H"
line_of(){ case "$1" in O1|O1_lora|K[0-9]*|RK[0-9]*|GK[0-9]*) echo offender;; C1|C1_lora|BEN|P[0-9]*|RP[0-9]*|D[0-9]*|GP[0-9]*|ref[AB][12]) echo clean;; *) echo UNKNOWN;; esac; }
for s in $ORDER; do
  d="$ROOT/${SES[$s]}/adapters"
  for m in $(ls "$d" | sort -V); do
    l=$(line_of "$m"); [ "$l" = UNKNOWN ] && fail "unknown model name $m in $s"
    for f in adapter_model.safetensors adapter_config.json; do
      [ -f "$d/$m/$f" ] && echo "$s,$m,$l,$f,$(sha256sum "$d/$m/$f" | cut -c1-64)" >> "$H"
    done
  done
  echo "  $s hashed"
done
for p in $(find "$REFS" -name adapter_model.safetensors | sort); do
  m=$(basename "$(dirname "$p")"); b=$(basename "$(dirname "$(dirname "$p")")")
  for f in adapter_model.safetensors adapter_config.json; do
    [ -f "$(dirname "$p")/$f" ] && echo "ref_$b,$m,clean,$f,$(sha256sum "$(dirname "$p")/$f" | cut -c1-64)" >> "$H"
  done
done
echo "  references hashed; $(($(wc -l < "$H")-1)) files in $H"

echo "== 4/4 packing (one tar per session, about 2-7 GB each)"
for s in $ORDER; do
  t="$OUT/p2_models_$s.tar"; [ -f "$t" ] && { echo "  $t exists, kept"; continue; }
  tar cf "$t.part" -C "$ROOT" "${SES[$s]}/adapters" && mv "$t.part" "$t" || fail "tar failed for $s"
  echo "  packed $t"
done
t="$OUT/p2_models_refs.tar"; [ -f "$t" ] || { tar cf "$t.part" -C "$ROOT" p2_v6/refs && mv "$t.part" "$t" || fail "tar failed for refs"; echo "  packed $t"; }
( cd "$OUT"; for f in p2_models_*.tar checkpoint_hashes_all.csv; do echo "$(stat -c %s "$f") $(md5sum "$f" | cut -c1-32) $f"; done ) | tee "$OUT/ARCHIVE_MANIFEST.txt"
echo "ARCHIVE READY: 7 lines above (6 tars + the hash list). Paste them to Claude."
