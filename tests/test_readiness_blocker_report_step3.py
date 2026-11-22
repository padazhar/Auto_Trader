from inference.readiness import readiness_blocker_report


def test_readiness_blocker_report_groups_model_registry_and_feature_reasons():
    readiness = {
        "ready_count": 1,
        "blocked_count": 2,
        "rows": [
            {
                "symbol": "AAAUSDT",
                "interval": "15m",
                "ready": False,
                "reason": "anchor_training_incomplete:rf:registry_not_execution_ready:active_without_registry_approval",
                "models": {
                    "rf": {
                        "ready": False,
                        "reason": "registry_not_execution_ready:active_without_registry_approval",
                        "status": "success",
                    },
                    "gbm": {"ready": True, "reason": "ready"},
                },
            },
            {
                "symbol": "BBBUSDT",
                "interval": "15m",
                "ready": False,
                "reason": "anchor_training_incomplete:gbm:feature_incompatible:feature_signature_mismatch",
                "models": {
                    "rf": {"ready": True, "reason": "ready"},
                    "gbm": {
                        "ready": False,
                        "reason": "feature_incompatible:feature_signature_mismatch",
                        "status": "success",
                    },
                },
            },
        ],
    }

    report = readiness_blocker_report(readiness)

    assert report["schema_version"] == "readiness_blocker_report.v1"
    assert report["total_blockers"] == 2
    assert report["summary"]["registry"] == 1
    assert report["summary"]["feature_compatibility"] == 1
    assert report["buckets"]["registry"][0]["symbol"] == "AAAUSDT"
    assert report["buckets"]["feature_compatibility"][0]["model_type"] == "gbm"


def test_readiness_blocker_report_includes_runtime_lifecycle_blockers():
    report = readiness_blocker_report(
        {"ready_count": 5, "blocked_count": 0, "rows": []},
        runtime_blockers=["active_universe_count_mismatch:5/10", "blocked_symbols:15"],
    )

    assert report["total_blockers"] == 2
    assert report["runtime_blocker_count"] == 2
    assert report["runtime_blockers"] == ["active_universe_count_mismatch:5/10", "blocked_symbols:15"]
    assert report["summary"]["symbol"] == 2
    assert report["buckets"]["symbol"][0]["source"] == "runtime"
