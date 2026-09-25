#!/usr/bin/env bash
# Run on the allocated Prefill host. Starts one TP=1 vLLM producers,
# per GPU in PREFILL_GPU_IDS (1-8 GPUs; endpoint i uses port base + i). Optional
# PREFILL_GPU_UUIDS (same order) is checked before each launch: GPUs are not
# cgroup-constrained on the cluster, so a wrong index would hit another job's GPU.
# Supply site-specific values through the variables below.
set -euo pipefail

: "${PYTHON_BIN:?Set PYTHON_BIN to the vLLM environment's Python executable}"
: "${MODEL_PATH:?Set MODEL_PATH to a local model snapshot directory}"
: "${PD_WORK_DIR:?Set PD_WORK_DIR to this job's writable work directory}"
: "${PREFILL_GPU_IDS:?Set PREFILL_GPU_IDS, for example 0,1,2,3}"
: "${PREFILL_HTTP_PORT_BASE:?Set PREFILL_HTTP_PORT_BASE, for example 8100}"
: "${PREFILL_KV_PORT_BASE:?Set PREFILL_KV_PORT_BASE, for example 14579}"
: "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME to the P-D network interface}"
PREFILL_KV_SEND_TYPE="${PREFILL_KV_SEND_TYPE:-PUT_ASYNC}"
[[ "$PREFILL_KV_SEND_TYPE" == PUT || "$PREFILL_KV_SEND_TYPE" == PUT_ASYNC ]] || {
  echo "PREFILL_KV_SEND_TYPE must be PUT or PUT_ASYNC" >&2; exit 2;
}

[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Model directory does not exist: $MODEL_PATH" >&2; exit 2; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }

IFS=, read -r -a gpu_ids <<< "$PREFILL_GPU_IDS"
(( ${#gpu_ids[@]} >= 1 && ${#gpu_ids[@]} <= 8 )) || { echo "1-8 Prefill GPU IDs are required" >&2; exit 2; }
gpu_uuids=()
if [[ -n "${PREFILL_GPU_UUIDS:-}" ]]; then
  IFS=, read -r -a gpu_uuids <<< "$PREFILL_GPU_UUIDS"
  [[ ${#gpu_uuids[@]} -eq ${#gpu_ids[@]} ]] || { echo "PREFILL_GPU_UUIDS must match PREFILL_GPU_IDS" >&2; exit 2; }
fi
for gpu_id in "${gpu_ids[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $gpu_id" >&2; exit 2; }
done
[[ ${#gpu_ids[@]} -eq $(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l | tr -d ' ') ]] || {
  echo "Prefill GPU IDs must be unique" >&2; exit 2;
}

ROLE_WORK_DIR="${PD_WORK_DIR}/prefill"
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

for index in "${!gpu_ids[@]}"; do
  gpu_id=${gpu_ids[$index]}
  if [[ ${#gpu_uuids[@]} -gt 0 ]]; then
    actual=$(nvidia-smi -i "$gpu_id" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [[ "$actual" == "${gpu_uuids[$index]}" ]] || {
      echo "GPU $gpu_id is $actual, expected ${gpu_uuids[$index]}; refusing to start" >&2; exit 97;
    }
  fi
  http_port=$((PREFILL_HTTP_PORT_BASE + index))
  kv_port=$((PREFILL_KV_PORT_BASE + index))
  kv_config=$(printf '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_port":%d,"kv_connector_extra_config":{"send_type":"%s"}}' "$kv_port" "$PREFILL_KV_SEND_TYPE")
  log_file="${ROLE_WORK_DIR}/logs/prefill_${index}.log"

  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --host 0.0.0.0 --port "$http_port" \
    --tensor-parallel-size 1 --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --kv-transfer-config "$kv_config" \
    --no-enable-prefix-caching --no-enable-chunked-prefill \
    --disable-log-requests >"$log_file" 2>&1 &
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
  [[ "$ready" -eq 1 ]] || { echo "P${index} failed to start; see $log_file" >&2; exit 1; }
  echo "P${index} ready: GPU=${gpu_id} HTTP=${http_port} KV=${kv_port}"
done

echo "All ${#gpu_ids[@]} Prefill instances are ready"
# Stay up while every instance runs (portable: no `wait -n`, which needs bash 4.3).
while true; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      echo "A Prefill instance exited unexpectedly (status $rc)" >&2
      exit 1
    fi
  done
  sleep 2
done
