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


def test_mixformerv2_block_normalizes_to_runtime_params():
    params = normalize_runtime_config({
        "mixformerv2": {
            "enabled": True,
            "checkpoint": "models/MixFormerV2/models/mixformerv2_small.pth.tar",
            "config_yaml": "models/MixFormerV2/experiments/mixformer2_vit_online/224_depth4_mlp1_score.yaml",
            "repo_root": "models/MixFormerV2",
            "search_factor": 4.55,
            "fusion": {
                "enabled": True,
                "weight": 0.4,
                "min_confidence": 0.25,
                "confidence_weighted": False,
            },
        },
    })

    assert params["mixformerv2_enabled"] is True
    assert params["mixformerv2_fusion_enabled"] is True
    assert params["mixformerv2_fusion_weight"] == pytest.approx(0.4)
    assert params["mixformerv2_fusion_min_confidence"] == pytest.approx(0.25)
    assert params["mixformerv2_fusion_confidence_weighted"] is False
    assert params["mixformerv2_search_factor"] == pytest.approx(4.55)
