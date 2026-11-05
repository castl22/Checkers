#!/bin/bash
# GPU tests of the writer: the restart round-trip regression test, then the
# save-failure test.  One GPU:  flux batch -N1 -q pdebug -t 15m tests/run_save_tests.sh <outdir>
set -uo pipefail
T=/usr/WS1/kogiou1/LLM_work/Checkers/tests
OUT=${1:?outdir}; mkdir -p "$OUT"
source /usr/workspace/kogiou1/venvs/deespeed_venv_tuo/bin/activate
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load rocm/6.4.0 2>/dev/null
export HIP_VISIBLE_DEVICES=0 DS_ACCELERATOR=cuda DS_SKIP_CUDA_CHECK=1 HIP_PLATFORM=amd
export HSA_OVERRIDE_GFX_VERSION=9.4.2 ROCM_PATH=/opt/rocm-6.4.0 CUDA_HOME=/opt/rocm-6.4.0
export TORCH_EXTENSIONS_DIR=/usr/workspace/kogiou1/.cache/torch_extensions
export CC=/usr/tce/bin/gcc CXX=/usr/tce/bin/g++
cd "$OUT"   # Checkers writes logs/<run>/rank_0/checkers.log under the cwd
rc=0
echo "##### regression: run_restart_roundtrip.sh"
bash "$T/run_restart_roundtrip.sh" "$OUT/roundtrip" || rc=1
echo "##### save failures: test_save_failures.py"
python "$T/test_save_failures.py" --out "$OUT/failures" > "$OUT/failures.log" 2>&1 || rc=1
grep -E "^===|^  (PASS|FAIL)|Traceback|Error" "$OUT/failures.log"
echo "##### ALL: $([ $rc -eq 0 ] && echo PASS || echo FAIL)"
exit $rc
