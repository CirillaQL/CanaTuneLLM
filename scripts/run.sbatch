#!/usr/bin/env bash
# Submit from the CanaTune repository root. Override these static Slurm defaults
# with sbatch command-line options when the site allocation differs.
#SBATCH --job-name=canatune
#SBATCH --partition=long
#SBATCH --time=16:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

die() {
  echo "run.sbatch: $*" >&2
  exit 1
}

: "${SLURM_JOB_ID:?Submit this file with sbatch}"
: "${SLURM_JOB_NODELIST:?Slurm did not provide an allocated node list}"
PROJECT_ROOT="${CANATUNE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export CANATUNE_CONFIG="${CANATUNE_CONFIG:-$PROJECT_ROOT/config.yaml}"
CANATUNE_PYTHON="${CANATUNE_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

[[ -f "$CANATUNE_CONFIG" ]] || die "Config file not found: $CANATUNE_CONFIG"
[[ -x "$CANATUNE_PYTHON" ]] || die "Install the project environment first: $CANATUNE_PYTHON"
[[ -r "$PROJECT_ROOT/scripts/start_4p_prefill_node.sh" ]] || die "Prefill script missing"
[[ -r "$PROJECT_ROOT/scripts/start_4d_decode_node.sh" ]] || die "Decode script missing"
command -v srun >/dev/null || die "srun is required"
command -v scontrol >/dev/null || die "scontrol is required"

# Read static topology and port settings from the same YAML used by the proxy.
mapfile -t settings < <("$CANATUNE_PYTHON" - "$CANATUNE_CONFIG" <<'PY'
import re
import sys

from canatune.config import load_config

config = load_config(sys.argv[1])
if config["slurm"]["nodes"] != 2 or config["container"]["enabled"]:
    raise SystemExit("run.sbatch requires a two-node non-container configuration")
if config["proxy"]["enabled"] is not True:
    raise SystemExit("proxy.enabled must be true")

topology = config["topology"]
prefill = topology["prefill_nodegroup"]
decode = topology["decode_nodegroup"]
values = (
    prefill["node"],
    decode["node"],
    config["runtime"]["python"],
    config["project"]["work_dir_root"],
    config["model"]["path_override_env"],
    config["model"]["max_model_len"],
    ",".join(map(str, prefill["gpu_ids"])),
    prefill["http_port_base"],
    prefill["kv_port_base"],
    ",".join(map(str, decode["gpu_ids"])),
    decode["http_port_base"],
    decode["kv_port_base"],
    decode["gpu_memory_utilization"],
    config["runtime"]["environment"]["NCCL_SOCKET_IFNAME"],
    config["kv_transfer"]["prefill_send_type"],
)
for value in values:
    text = str(value)
    if not text or re.search(r"[\r\n]", text):
        raise SystemExit("configuration value is empty or contains a newline")
    print(text)
PY
)
[[ ${#settings[@]} -eq 15 ]] || die "Could not read the expected settings from config.yaml"

PREFILL_NODE="${PREFILL_NODE:-${settings[0]}}"
DECODE_NODE="${DECODE_NODE:-${settings[1]}}"
[[ "$PREFILL_NODE" != "$DECODE_NODE" ]] || die "Prefill and Decode need different nodes"
PYTHON_BIN="${PYTHON_BIN:-${settings[2]}}"
PD_WORK_DIR="${PD_WORK_DIR:-${settings[3]}/${SLURM_JOB_ID}}"
model_override_name="${settings[4]}"
[[ "$model_override_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "Invalid model override variable"
model_override_value="${!model_override_name-}"
MODEL_PATH="${MODEL_PATH:-$model_override_value}"
: "${MODEL_PATH:?Set MODEL_PATH to the model snapshot directory}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${settings[5]}}"
PREFILL_GPU_IDS="${PREFILL_GPU_IDS:-${settings[6]}}"
PREFILL_HTTP_PORT_BASE="${PREFILL_HTTP_PORT_BASE:-${settings[7]}}"
PREFILL_KV_PORT_BASE="${PREFILL_KV_PORT_BASE:-${settings[8]}}"
DECODE_GPU_IDS="${DECODE_GPU_IDS:-${settings[9]}}"
DECODE_HTTP_PORT_BASE="${DECODE_HTTP_PORT_BASE:-${settings[10]}}"
DECODE_KV_PORT_BASE="${DECODE_KV_PORT_BASE:-${settings[11]}}"
DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-${settings[12]}}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${settings[13]}}"
PREFILL_KV_SEND_TYPE="${PREFILL_KV_SEND_TYPE:-${settings[14]}}"

export PYTHON_BIN MODEL_PATH PD_WORK_DIR MAX_MODEL_LEN
export PREFILL_GPU_IDS PREFILL_HTTP_PORT_BASE PREFILL_KV_PORT_BASE
export DECODE_GPU_IDS DECODE_HTTP_PORT_BASE DECODE_KV_PORT_BASE
export DECODE_GPU_MEMORY_UTILIZATION NCCL_SOCKET_IFNAME PREFILL_KV_SEND_TYPE
export PREFILL_NODE DECODE_NODE

mapfile -t allocated_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#allocated_nodes[@]} -eq 2 ]] || die "Exactly two allocated nodes are required"
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
  [[ " ${allocated_nodes[*]} " == *" $node "* ]] || die "$node is not in the allocation"
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" bash -c \
    '[[ -x "$PYTHON_BIN" && -f "$MODEL_PATH/config.json" && -r "$PROJECT_ROOT/scripts/start_4p_prefill_node.sh" && -r "$PROJECT_ROOT/scripts/start_4d_decode_node.sh" ]]' \
    || die "Python, model, or project scripts are unavailable on $node"
done

resolve_ip() {
  local addresses ip ignored
  addresses=$(srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" hostname -I) \
    || die "Cannot resolve $1"
  read -r ip ignored <<< "$addresses"
  [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || die "The first hostname -I address on $1 must be IPv4"
  printf '%s\n' "$ip"
}

CANATUNE_PREFILL_HOST="${CANATUNE_PREFILL_HOST:-$(resolve_ip "$PREFILL_NODE")}"
CANATUNE_DECODE_HOST="${CANATUNE_DECODE_HOST:-$(resolve_ip "$DECODE_NODE")}"
export CANATUNE_PREFILL_HOST CANATUNE_DECODE_HOST
export CANATUNE_PREFILL_KV_HOST="${CANATUNE_PREFILL_KV_HOST:-$CANATUNE_PREFILL_HOST}"
export CANATUNE_DECODE_KV_HOST="${CANATUNE_DECODE_KV_HOST:-$CANATUNE_DECODE_HOST}"

# Replace the batch shell so the Python controller receives Slurm's stop signal.
exec "$CANATUNE_PYTHON" -m canatune.controller.process_controller
