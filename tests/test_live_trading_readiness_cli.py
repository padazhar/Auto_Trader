from __future__ import annotations

from pathlib import Path

from scripts import live_trading_readiness_check as module


def test_live_trading_readiness_main_defaults_to_cwd(monkeypatch, tmp_path: Path) -> None:
    called = {}

    class _DummyReport:
        live_trading_ready = False

        def to_dict(self):
            return {"live_trading_ready": False, "overall_status": "fail", "items": [], "project_root": str(tmp_path)}

    monkeypatch.chdir(tmp_path)
    def _fake_run(root):
        called["root"] = root
        return _DummyReport()

    monkeypatch.setattr(module, "run_live_trading_readiness_check", _fake_run)
    monkeypatch.setattr("sys.argv", ["live_trading_readiness_check"])

    exit_code = module.main()

    assert exit_code == 1
    assert Path(called["root"]).resolve() == tmp_path.resolve()
