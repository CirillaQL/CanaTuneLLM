#!/usr/bin/env bash
# Run on the allocated Decode host. Starts one vLLM consumer per
# DECODE_TP GPUs of DECODE_GPU_IDS (1-8 GPUs; ports below). Optional
# DECODE_GPU_UUIDS (same order) is checked before each launch: GPUs are not
# cgroup-constrained on the cluster, so a wrong index would hit another job's GPU.
# KV_CONNECTOR: NixlConnector (default; D reads P's KV, endpoint i's own side channel
# on its KV port, below) or P2pNcclConnector.
# Supply site-specific values through the variables below.
# Tensor parallelism: DECODE_TP (default 1) consecutive GPUs of DECODE_GPU_IDS form one
# instance (--tensor-parallel-size); instance i uses HTTP port base + i and KV port
# base + i x TP (vLLM's connector rank r listens on kv_port + r). A role on several
# nodes runs this script once per node: DECODE_INDEX_OFFSET numbers its instances after
# the earlier nodes' (names, logs) and DECODE_ROLE_DIR (default decode) separates its
# work directory on the shared file system.
set -euo pipefail

: "${PYTHON_BIN:?Set PYTHON_BIN to the vLLM environment's Python executable}"
: "${MODEL_PATH:?Set MODEL_PATH to a local model snapshot directory}"
: "${PD_WORK_DIR:?Set PD_WORK_DIR to this job's writable work directory}"
: "${DECODE_GPU_IDS:?Set DECODE_GPU_IDS, for example 0,1,2,3}"
: "${DECODE_HTTP_PORT_BASE:?Set DECODE_HTTP_PORT_BASE, for example 8200}"
: "${DECODE_KV_PORT_BASE:?Set DECODE_KV_PORT_BASE, for example 14579}"
KV_CONNECTOR="${KV_CONNECTOR:-NixlConnector}"
TP="${DECODE_TP:-1}"
INDEX_OFFSET="${DECODE_INDEX_OFFSET:-0}"
[[ "$TP" =~ ^[1-9][0-9]*$ && "$INDEX_OFFSET" =~ ^[0-9]+$ ]] || {
  echo "DECODE_TP must be a positive integer and DECODE_INDEX_OFFSET a non-negative one" >&2; exit 2;
}
KV_LOAD_FAILURE_POLICY="${KV_LOAD_FAILURE_POLICY:-fail}"
KV_BUFFER_DEVICE="${KV_BUFFER_DEVICE:-cuda}"
[[ "$KV_BUFFER_DEVICE" == cuda || "$KV_BUFFER_DEVICE" == cpu ]] || {
  echo "KV_BUFFER_DEVICE must be cuda or cpu" >&2; exit 2;
}
if [[ "$KV_CONNECTOR" == P2pNcclConnector ]]; then
  : "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME to the P-D network interface}"
elif [[ "$KV_CONNECTOR" != NixlConnector ]]; then
  echo "KV_CONNECTOR must be NixlConnector or P2pNcclConnector" >&2; exit 2
fi
[[ "$KV_LOAD_FAILURE_POLICY" == fail || "$KV_LOAD_FAILURE_POLICY" == recompute ]] || {
  echo "KV_LOAD_FAILURE_POLICY must be fail or recompute" >&2; exit 2;
}
side_host="${CANATUNE_DECODE_KV_HOST:-${CANATUNE_DECODE_HOST:-}}"
[[ -n "$side_host" ]] || side_host=$(hostname -I | awk '{print $1}')

[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Model directory does not exist: $MODEL_PATH" >&2; exit 2; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }

IFS=, read -r -a gpu_ids <<< "$DECODE_GPU_IDS"
(( ${#gpu_ids[@]} >= 1 && ${#gpu_ids[@]} <= 8 )) || { echo "1-8 Decode GPU IDs are required" >&2; exit 2; }
gpu_uuids=()
if [[ -n "${DECODE_GPU_UUIDS:-}" ]]; then
  IFS=, read -r -a gpu_uuids <<< "$DECODE_GPU_UUIDS"
  [[ ${#gpu_uuids[@]} -eq ${#gpu_ids[@]} ]] || { echo "DECODE_GPU_UUIDS must match DECODE_GPU_IDS" >&2; exit 2; }
fi
for gpu_id in "${gpu_ids[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $gpu_id" >&2; exit 2; }
done
(( ${#gpu_ids[@]} % TP == 0 )) || { echo "DECODE_GPU_IDS must be a multiple of DECODE_TP=$TP" >&2; exit 2; }
[[ ${#gpu_ids[@]} -eq $(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l | tr -d ' ') ]] || {
  echo "Decode GPU IDs must be unique" >&2; exit 2;
}

ROLE_WORK_DIR="${PD_WORK_DIR}/${DECODE_ROLE_DIR:-decode}"
export HOME="${ROLE_WORK_DIR}/home"
export HF_HOME="${ROLE_WORK_DIR}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="${ROLE_WORK_DIR}/xdg-cache"
export XDG_CONFIG_HOME="${ROLE_WORK_DIR}/xdg-config"
export XDG_DATA_HOME="${ROLE_WORK_DIR}/xdg-data"
export XDG_STATE_HOME="${ROLE_WORK_DIR}/xdg-state"
export XDG_RUNTIME_DIR="${ROLE_WORK_DIR}/xdg-runtime"
export VLLM_CACHE_ROOT="${ROLE_WORK_DIR}/vllm-cache"
export VLLM_CONFIG_ROOT="${ROLE_WORK_DIR}/vllm-config"
export TORCH_HOME="${ROLE_WORK_DIR}/torch-cache"
export TRITON_CACHE_DIR="${ROLE_WORK_DIR}/triton-cache"
export CUDA_CACHE_PATH="${ROLE_WORK_DIR}/cuda-cache"
export TMPDIR="${ROLE_WORK_DIR}/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_RUNTIME_DIR" "$VLLM_CACHE_ROOT" \
  "$VLLM_CONFIG_ROOT" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$TMPDIR" "${ROLE_WORK_DIR}/logs"
chmod 700 "$XDG_RUNTIME_DIR" "$TMPDIR"

pids=()
cleanup() {
  trap - EXIT INT TERM
  local pid alive deadline
  local grace="${NODE_SHUTDOWN_GRACE_S:-20}"
  [[ "$grace" =~ ^[1-9][0-9]*$ ]] || grace=20
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  deadline=$((SECONDS + grace))
  while ((SECONDS < deadline)); do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=1; break; fi
    done
    ((alive == 0)) && break
    sleep 0.2
  done
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((index = 0; index < ${#gpu_ids[@]} / TP; index++)); do
  name=$((INDEX_OFFSET + index))
  devices=""
  for ((rank = 0; rank < TP; rank++)); do
    slot=$((index * TP + rank))
    gpu_id=${gpu_ids[$slot]}
    if [[ ${#gpu_uuids[@]} -gt 0 ]]; then
      actual=$(nvidia-smi -i "$gpu_id" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
      [[ "$actual" == "${gpu_uuids[$slot]}" ]] || {
        echo "GPU $gpu_id is $actual, expected ${gpu_uuids[$slot]}; refusing to start" >&2; exit 97;
      }
    fi
    devices="${devices:+$devices,}$gpu_id"
  done
  http_port=$((DECODE_HTTP_PORT_BASE + index))
  kv_port=$((DECODE_KV_PORT_BASE + index * TP))
  if [[ "$KV_CONNECTOR" == NixlConnector ]]; then
    kv_config=$(printf '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"%s","kv_buffer_device":"%s"}' "$KV_LOAD_FAILURE_POLICY" "$KV_BUFFER_DEVICE")
    kv_env=("VLLM_NIXL_SIDE_CHANNEL_HOST=$side_host" "VLLM_NIXL_SIDE_CHANNEL_PORT=$kv_port")
  else
    kv_config=$(printf '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_port":%d}' "$kv_port")
    kv_env=()
  fi
  # GPU memory for vLLM: a fixed DECODE_GPU_MEMORY_UTILIZATION, or (unset / auto) the
  # GPU's memory less the configured reserve (canatune.infrastructure.memory_plan).
  mem_util="${DECODE_GPU_MEMORY_UTILIZATION:-auto}"
  if [[ "$mem_util" == auto ]]; then
    : "${CANATUNE_CONFIG:?Set CANATUNE_CONFIG (or a fixed DECODE_GPU_MEMORY_UTILIZATION)}"
    total_mib=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
    mem_util=$(PYTHONPATH="${PROJECT_ROOT:+$PROJECT_ROOT/src:}${PYTHONPATH:-}" "$PYTHON_BIN" \
      -m canatune.infrastructure.memory_plan --config "$CANATUNE_CONFIG" --total-mib "$total_mib") \
      || { echo "decode_${name}: no memory plan for GPU $gpu_id" >&2; exit 2; }
  fi
  echo "decode_${name}: GPU $devices gpu-memory-utilization $mem_util"
  log_file="${ROLE_WORK_DIR}/logs/decode_${name}.log"

  # Optional: a chat template for /v1/chat/completions (a base model has none);
  # P and D must use the same one so both see the same prompt tokens.
  chat_args=()
  if [[ -n "${VLLM_CHAT_TEMPLATE:-}" ]]; then chat_args=(--chat-template "$VLLM_CHAT_TEMPLATE"); fi
  CUDA_VISIBLE_DEVICES="$devices" env ${kv_env[@]+"${kv_env[@]}"} \
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --host 0.0.0.0 --port "$http_port" \
    --tensor-parallel-size "$TP" --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization "$mem_util" \
    --kv-transfer-config "$kv_config" \
    --no-enable-prefix-caching --no-enable-chunked-prefill \
    ${chat_args[@]+"${chat_args[@]}"} >"$log_file" 2>&1 &
  pids+=("$!")

  ready=0
  for ((attempt=0; attempt<${STARTUP_TIMEOUT_S:-1500}; attempt+=5)); do
    if ! kill -0 "${pids[${#pids[@]}-1]}" 2>/dev/null; then break; fi
    if curl -fsS --connect-timeout 2 --max-time 3 "http://127.0.0.1:${http_port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  [[ "$ready" -eq 1 ]] || { echo "D${name} failed to start; see $log_file" >&2; exit 1; }
  echo "D${name} ready: GPU=${devices} HTTP=${http_port} KV=${kv_port}"
done

echo "All $(( ${#gpu_ids[@]} / TP )) Decode instances are ready (TP $TP)"
# Stay up while every instance runs (portable: no `wait -n`, which needs bash 4.3).
while true; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      echo "A Decode instance exited unexpectedly (status $rc)" >&2
      exit 1
    fi
  done
  sleep 2
done
