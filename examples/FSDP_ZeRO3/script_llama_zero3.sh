#!/bin/bash

# 1. Environment Setup
source /usr/workspace/kogiou1/venvs/deespeed_venv_tuo/bin/activate
module load rocm/6.4.0

# 1. Clear the failed build again
rm -rf /g/g92/kogiou1/.cache/torch_extensions/

# 2. Set explicit paths for BOTH the compiler and the C++ wrapper
export CC=$(which gcc)
export CXX=$(which g++)
export PATH=$(dirname $(which gcc)):$PATH
export PATH=$(python -c "import sys; import os; print(os.path.dirname(sys.executable))"):$PATH
export PATH=/usr/workspace/kogiou1/venvs/deespeed_venv_tuo/bin:$PATH
export CMAKE_BIN_DIR=/usr/workspace/kogiou1/venvs/deespeed_venv_tuo/bin
export PYTHONPATH=$PYTHONPATH:/usr/WS1/kogiou1/LLM_work/Checkers/build

# 2. THE CRITICAL FIXES FOR AMD
export DS_ACCELERATOR=cuda
export DS_SKIP_CUDA_CHECK=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export HIP_PLATFORM=amd
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export ROCM_PATH=/opt/rocm-6.4.0
export CUDA_HOME=$ROCM_PATH

# 3. Path Overrides (Crucial for Lustre usage)
export HF_HOME="/p/vast1/kogiou1/hf_cache"
export HF_DATASETS_CACHE="/p/vast1/kogiou1/hf_datasets"
export HF_DATASETS_TRUST_REMOTE_CODE=True


# 4. Generate Hostfile for 2 nodes, 4 slots each
flux run -N 2 -n 8 hostname | sort -u | awk '{print $1 " slots=4"}' > dp_llama_hostfile

echo "--- Generated Hostfile ---"
cat dp_llama_hostfile
echo "--------------------------"

# 5. Network Setup
export MASTER_ADDR=$(head -n 1 dp_llama_hostfile | awk '{print $1}')
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# 6. Launch Training
# Using your requested parameters: 2 nodes, 4 GPUs per node
deepspeed \
    --launcher pdsh \
    --hostfile dp_llama_hostfile \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --num_nodes 2 \
    --num_gpus 4 \
    train_llama3.py --deepspeed_config ds_config_zero3.json