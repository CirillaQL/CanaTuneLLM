"""Room's fallback and the real Decode launch command share one engine limit."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_router_controller import room_setup

from canatune.config import ConfigError, decode_max_num_seqs, load_config
from canatune.controller.router import RouterSettings


@pytest.mark.parametrize("limit", [0, -1, True, None, 1.5, "512"])
def test_invalid_decode_engine_limit_is_rejected(limit):
    config = load_config()
    config["runtime"]["decode_max_num_seqs"] = limit
    with pytest.raises(ConfigError, match="positive integer"):
        RouterSettings.from_config(config)


def test_old_null_room_table_uses_engine_cap_and_counts_first_token_reservations():
    _, groups, router, _ = room_setup()
    router._apply_table()
    assert router.room_d_sequences == 256
    groups[1].n_decoding, groups[1].n_await = 250, 5
    assert router.has_room(groups[1])  # one more fits
    groups[1].n_await += 1
    assert not router.has_room(groups[1]) and router.has_room(groups[2])


def test_null_room_limit_spills_at_custom_engine_cap():
    _, _, router, _ = room_setup()
    router.settings = dataclasses.replace(router.settings, decode_max_num_seqs=3)
    tickets = [router.try_admit(128, True, best_effort=True) for _ in range(3)]
    assert {t.group.name for t in tickets} == {"G1"}
    assert router.try_admit(128, True, best_effort=True).group.name == "G2"


@pytest.mark.parametrize("measured,expected", [(5, 5), (20, 8), (None, 8)])
def test_measured_room_cap_cannot_exceed_configured_engine_cap(measured, expected):
    _, _, router, _ = room_setup(d_sequences=measured)
    router.settings = dataclasses.replace(router.settings, decode_max_num_seqs=8)
    router._apply_table()
    assert router.room_d_sequences == expected


@pytest.mark.parametrize("configured", [None, 64])
def test_decode_launcher_passes_the_same_limit_as_router(tmp_path, configured):
    root = Path(__file__).resolve().parents[1]
    config = load_config()
    if configured is None:
        del config["runtime"]["decode_max_num_seqs"]  # old config keeps the A100 default
    else:
        config["runtime"]["decode_max_num_seqs"] = configured
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))  # JSON is valid YAML
    model = tmp_path / "model"
    model.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" == -m && "$2" == vllm.entrypoints.openai.api_server ]]; then\n'
        '  printf "%s\\n" "$@" > "$TRACE"\n'
        '  "$REAL_PYTHON" -c "import time; time.sleep(0.3)"\n'
        'else exec "$REAL_PYTHON" "$@"; fi\n'
    )
    curl = bin_dir / "curl"
    curl.write_text("#!/usr/bin/env bash\nexit 0\n")
    for file in (fake_python, curl):
        file.chmod(0o755)
    trace = tmp_path / "argv"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "PYTHON_BIN": str(fake_python), "REAL_PYTHON": sys.executable,
           "PROJECT_ROOT": str(root), "CANATUNE_CONFIG": str(path),
           "MODEL_PATH": str(model), "PD_WORK_DIR": str(tmp_path / "work"),
           "DECODE_GPU_IDS": "0", "DECODE_GPU_UUIDS": "", "TRACE": str(trace),
           "DECODE_HTTP_PORT_BASE": "8200", "DECODE_KV_PORT_BASE": "14579",
           "CANATUNE_DECODE_HOST": "127.0.0.1", "KV_CONNECTOR": "NixlConnector",
           "KV_BUFFER_DEVICE": "cuda", "KV_LOAD_FAILURE_POLICY": "fail",
           "DECODE_GPU_MEMORY_UTILIZATION": "0.95"}
    result = subprocess.run(["bash", str(root / "scripts/start_4d_decode_node.sh")],
                            env=env, text=True, capture_output=True, timeout=10)
    # The stub exits after recording argv; the supervisor must report its exit.
    assert result.returncode == 1, result.stdout + result.stderr
    assert "All 1 Decode instances are ready" in result.stdout
    assert "exited unexpectedly (status 0)" in result.stderr
    args = trace.read_text().splitlines()
    limit = int(args[args.index("--max-num-seqs") + 1])
    assert limit == decode_max_num_seqs(config)
    assert limit == RouterSettings.from_config(config).decode_max_num_seqs
