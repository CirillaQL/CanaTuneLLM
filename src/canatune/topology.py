"""Endpoint placement: which node and GPUs serve each endpoint.

An endpoint is one vLLM instance. With tensor parallelism it spans
`tensor_parallel_size` GPUs of one node (TP never crosses nodes: no NVLink, and an
all-reduce per layer over Ethernet would dominate); a role may span several nodes.

  topology.<role>_nodegroup.tensor_parallel_size   GPUs per endpoint (default 1)
  topology.<role>_nodegroup.nodes                  [{node, gpu_ids, gpu_uuids}] per node;
                                                   absent: the group's node / gpu_ids
  topology.endpoints.<name>.node / gpu_ids         its node and GPUs (absent: the role's
                                                   node and [gpu_id])

Hosts: CANATUNE_NODE_HOSTS ("node=ip,node=ip") maps node names to addresses; without
an entry an endpoint uses its role's host (CANATUNE_PREFILL_HOST / CANATUNE_DECODE_HOST,
else the node name).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from canatune.config import P2P_NCCL_CONNECTOR, ConfigError, kv_connector

ROLE_HOST_ENV = {"prefill": "CANATUNE_PREFILL_HOST", "decode": "CANATUNE_DECODE_HOST"}


def tensor_parallel_size(config: Mapping[str, Any], role: str) -> int:
    group = config.get("topology", {}).get(f"{role}_nodegroup") or {}
    value = group.get("tensor_parallel_size", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{role}_nodegroup.tensor_parallel_size must be a positive integer")
    return value


def check_tensor_parallel(config: Mapping[str, Any]) -> None:
    """P2pNcclConnector pairs TP rank r of P with rank r of D (vLLM: symmetric TP only)."""
    p, d = tensor_parallel_size(config, "prefill"), tensor_parallel_size(config, "decode")
    if p != d and kv_connector(config) == P2P_NCCL_CONNECTOR:
        raise ConfigError(f"P2pNcclConnector needs the same TP on P and D (P {p}, D {d})")


def role_nodes(config: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    """[{node, gpu_ids, gpu_uuids}] of a role, one entry per node."""
    group = config["topology"][f"{role}_nodegroup"]
    nodes = group.get("nodes")
    if nodes:
        return [
            {
                "node": str(n["node"]),
                "gpu_ids": [int(g) for g in n["gpu_ids"]],
                "gpu_uuids": n.get("gpu_uuids"),
            }
            for n in nodes
        ]
    return [
        {
            "node": str(group["node"]),
            "gpu_ids": [int(g) for g in group.get("gpu_ids") or []],
            "gpu_uuids": group.get("gpu_uuids"),
        }
    ]


def endpoint_gpus(endpoint: Mapping[str, Any]) -> tuple[int, ...]:
    """The GPU indices of one endpoint (its node's numbering)."""
    if endpoint.get("gpu_ids") is not None:
        return tuple(int(g) for g in endpoint["gpu_ids"])
    return (int(endpoint["gpu_id"]),)


def endpoint_node(config: Mapping[str, Any], endpoint: Mapping[str, Any]) -> str:
    node = endpoint.get("node")
    if node:
        return str(node)
    return str(config["topology"][f"{endpoint['role']}_nodegroup"].get("node", ""))


def node_hosts(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """CANATUNE_NODE_HOSTS: node name -> address."""
    raw = (os.environ if env is None else env).get("CANATUNE_NODE_HOSTS", "")
    out = {}
    for item in raw.split(","):
        name, sep, host = item.strip().partition("=")
        if sep and name and host:
            out[name] = host
    return out


def host_of(
    config: Mapping[str, Any], role: str, node: str, env: Mapping[str, str] | None = None
) -> str:
    env = os.environ if env is None else env
    hosts = node_hosts(env)
    if node in hosts:
        return hosts[node]
    return env.get(ROLE_HOST_ENV[role]) or node


def agent_port(config: Mapping[str, Any], role: str) -> int:
    raw = config.get("clock_control", {}).get("agent_port", 9300)
    return int(raw[role] if isinstance(raw, dict) else raw)


def agent_urls(
    config: Mapping[str, Any], env: Mapping[str, str] | None = None
) -> dict[tuple[str, str], str]:
    """(role, node) -> the GPU agent URL on that node."""
    out = {}
    for role in ("prefill", "decode"):
        for n in role_nodes(config, role):
            host = host_of(config, role, n["node"], env)
            out[(role, n["node"])] = f"http://{host}:{agent_port(config, role)}"
    return out
