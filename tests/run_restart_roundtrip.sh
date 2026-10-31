#!/bin/bash
# Regression test: Checkers save -> restore round trip with measure_snr on and off.
# Needs one GPU:   flux run -N1 -n1 -q pdebug -t 15m tests/run_restart_roundtrip.sh [outdir]
#
# PASS requires
#   1. test_restart_roundtrip.py passes with measure_snr=0 AND measure_snr=1
#   2. model_state / master_weights / exp_avg_sq .cch files are byte-identical
#      between the two (measure_snr is diagnostics only and must not change
#      what is written).  exp_avg is excluded: its TT-SVD is randomized.
set -uo pipefail
T=/usr/WS1/kogiou1/LLM_work/Checkers/tests   # absolute: flux batch runs a copy of this script from /var/tmp
OUT=${1:-$(mktemp -d /tmp/checkers_rt.XXXXXX)}
source /usr/workspace/kogiou1/venvs/deespeed_venv_tuo/bin/activate
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load rocm/6.4.0 2>/dev/null
export HIP_VISIBLE_DEVICES=0 DS_ACCELERATOR=cuda DS_SKIP_CUDA_CHECK=1 HIP_PLATFORM=amd
export HSA_OVERRIDE_GFX_VERSION=9.4.2 ROCM_PATH=/opt/rocm-6.4.0 CUDA_HOME=/opt/rocm-6.4.0
export TORCH_EXTENSIONS_DIR=/usr/workspace/kogiou1/.cache/torch_extensions
export CC=/usr/tce/bin/gcc CXX=/usr/tce/bin/g++

rc=0
for m in 0 1; do
  rm -rf "$OUT/snr$m"; mkdir -p "$OUT"
  python "$T/test_restart_roundtrip.py" --measure-snr $m --out "$OUT/snr$m" > "$OUT/snr$m.log" 2>&1
  prc=$?
  grep -E "^===|^  (PASS|FAIL)|Traceback|Error" "$OUT/snr$m.log"
  [ $prc -eq 0 ] || { echo "  (measure_snr=$m exited $prc; full log $OUT/snr$m.log)"; rc=1; }
done

echo "=== .cch byte-identity across measure_snr (per-group streams)"
for s in model_state master_weights exp_avg_sq; do
  a=$(md5sum < "$OUT/snr0/rank_000_$s.cch" 2>/dev/null | cut -d' ' -f1)
  b=$(md5sum < "$OUT/snr1/rank_000_$s.cch" 2>/dev/null | cut -d' ' -f1)
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "  PASS  $s identical"; else echo "  FAIL  $s differs ($a vs $b)"; rc=1; fi
done
echo "=== OVERALL: $([ $rc -eq 0 ] && echo PASS || echo FAIL)   (outputs in $OUT)"
exit $rc
