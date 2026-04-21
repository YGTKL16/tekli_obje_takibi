"""Tests for runtime config normalization."""

import pytest

from tracker.config import load_runtime_config, normalize_runtime_config


def test_canonical_conf_threshold_drives_decision_and_state_machine():
    params = normalize_runtime_config({
        "decision": {
            "confidence_threshold": 0.18,
            "conf_bypass_threshold": 0.15,
        },
        "smart_intervention": {
            "sm_conf_threshold": 0.42,
        },
    })

    assert params["conf_threshold"] == pytest.approx(0.18)
    assert params["sm_conf_threshold"] == pytest.approx(0.18)


def test_legacy_decision_conf_threshold_still_falls_back_cleanly():
    params = normalize_runtime_config({
        "decision": {
            "conf_threshold": 0.21,
        },
    })

    assert params["conf_threshold"] == pytest.approx(0.21)
    assert params["sm_conf_threshold"] == pytest.approx(0.21)


def test_hygiene_conf_threshold_falls_back_when_decision_missing():
    params = normalize_runtime_config({
        "hygiene": {
            "conf_threshold": 0.24,
        },
    })

    assert params["conf_threshold"] == pytest.approx(0.24)
    assert params["sm_conf_threshold"] == pytest.approx(0.24)


def test_canonical_bypass_threshold_is_preserved_from_decision():
    params = normalize_runtime_config({
        "decision": {
            "confidence_threshold": 0.18,
            "conf_bypass_threshold": 0.33,
        },
    })

    assert params["conf_bypass_threshold"] == pytest.approx(0.33)


def test_load_runtime_config_reads_yaml_file(tmp_path):
    config_path = tmp_path / "tracker_config.yaml"
    config_path.write_text(
        "decision:\n"
        "  confidence_threshold: 0.18\n"
        "  conf_bypass_threshold: 0.15\n",
        encoding="utf-8",
    )

    params = load_runtime_config(str(config_path))

    assert params["conf_threshold"] == pytest.approx(0.18)
    assert params["sm_conf_threshold"] == pytest.approx(0.18)
    assert params["conf_bypass_threshold"] == pytest.approx(0.15)
