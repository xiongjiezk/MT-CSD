#!/bin/bash

set -euo pipefail

# 可通过外部环境变量覆盖：CUDA_VISIBLE_DEVICES=0,1 ./run_train.sh config/config_sitpcl.yaml
: "${CUDA_VISIBLE_DEVICES:=0,1}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

config_file=${1:-}
if [ -z "$config_file" ]; then
    echo "Usage: ./run_train.sh <config_file_path>"
    exit 1
fi

if [ ! -f "$config_file" ]; then
    echo "Error: config file '$config_file' does not exist"
    exit 1
fi

mkdir -p logs

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="logs/train_${timestamp}.log"

# 根据可见 GPU 数自动决定启动方式：单卡直接 python，多卡使用 torchrun
nproc=$(python - <<'PY'
import os
v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(len([x for x in v.split(",") if x.strip()]) if v else 1)
PY
)

if [ "${nproc}" -le 1 ]; then
    cmd=(python main_12.py "${config_file}")
else
    cmd=(torchrun --standalone --nproc_per_node="${nproc}" main_12.py "${config_file}")
fi

setsid nohup "${cmd[@]}" > "${logfile}" 2>&1 < /dev/null &
train_pid=$!

echo "Training started, PID: ${train_pid}"
echo "Log file: ${logfile}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "nproc_per_node=${nproc}"
echo "Launch command: ${cmd[*]}"

tail -f "${logfile}" &
tail_pid=$!

trap '
  if ps -p ${train_pid} > /dev/null; then
    echo "Log tail stopped, training still running (PID: ${train_pid})."
  else
    echo "Training finished."
  fi
  kill ${tail_pid} 2>/dev/null || true
  trap - INT
  exit 0
' INT

wait ${tail_pid} || true
