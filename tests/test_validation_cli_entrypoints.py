from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script_name: str) -> tuple[int, dict]:
    result = subprocess.run(
        [sys.executable, f"scripts/{script_name}", "--project-root", "."],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, json.loads(result.stdout)


def test_repository_readiness_check_runs_as_direct_script() -> None:
    returncode, payload = _run_script("repository_readiness_check.py")
    assert returncode == 0
    assert payload["overall_status"] == "pass"
    assert payload["paper_trading_ready"] is True


def test_live_trading_readiness_check_runs_as_direct_script() -> None:
    returncode, payload = _run_script("live_trading_readiness_check.py")
    assert returncode == 1
    assert payload["overall_status"] == "fail"
    assert payload["live_trading_ready"] is False


def test_final_live_validation_gate_runs_as_direct_script() -> None:
    returncode, payload = _run_script("final_live_validation_gate.py")
    assert returncode == 1
    assert payload["overall_status"] == "warn"
    assert payload["controlled_live_candidate_ready"] is False
    assert payload["live_trading_ready"] is False
