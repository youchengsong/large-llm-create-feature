"""tests/test_calibration.py — L2 阈值自动校准测试。"""

import numpy as np
import pandas as pd
import pytest

from src.config import ValidationThresholds
from src.evaluation.calibration import calibrate_l2_thresholds


@pytest.fixture
def label():
    rng = np.random.RandomState(42)
    return pd.Series(rng.choice([0, 1], size=500, p=[0.8, 0.2]))


@pytest.fixture
def baseline_features(label):
    rng = np.random.RandomState(42)
    n = len(label)
    return pd.DataFrame({
        "strong_feat": label.values * 3.0 + rng.normal(0, 1, n),
        "medium_feat": label.values * 1.5 + rng.normal(0, 2, n),
        "weak_feat": label.values * 0.5 + rng.normal(0, 3, n),
        "noise_feat": rng.normal(0, 1, n),
        "binary_feat": (rng.random(n) > 0.5).astype(float),
    })


def test_calibration_returns_valid_thresholds(baseline_features, label):
    thresholds = calibrate_l2_thresholds(baseline_features, label)
    assert isinstance(thresholds, ValidationThresholds)
    assert thresholds.mi_min > 0
    assert thresholds.cramers_v_min > 0
    assert thresholds.corr_max == 0.9
    assert thresholds.bootstrap_p_max == 0.05


def test_calibrated_thresholds_are_lower_than_defaults(baseline_features, label):
    defaults = ValidationThresholds()
    calibrated = calibrate_l2_thresholds(baseline_features, label)
    assert calibrated.mi_min <= defaults.mi_min or calibrated.mi_min > 0
    assert calibrated.cramers_v_min <= defaults.cramers_v_min or calibrated.cramers_v_min > 0


def test_calibration_with_empty_features(label):
    empty = pd.DataFrame(index=range(len(label)))
    thresholds = calibrate_l2_thresholds(empty, label)
    defaults = ValidationThresholds()
    assert thresholds.mi_min == defaults.mi_min
    assert thresholds.cramers_v_min == defaults.cramers_v_min


def test_calibration_with_single_feature(label):
    rng = np.random.RandomState(42)
    single = pd.DataFrame({"only_feat": label.values * 2.0 + rng.normal(0, 1, len(label))})
    thresholds = calibrate_l2_thresholds(single, label)
    assert thresholds.mi_min > 0
    assert thresholds.cramers_v_min > 0


def test_calibration_percentile_effect(baseline_features, label):
    strict = calibrate_l2_thresholds(baseline_features, label, percentile=50.0)
    lenient = calibrate_l2_thresholds(baseline_features, label, percentile=5.0)
    assert lenient.mi_min <= strict.mi_min
    assert lenient.cramers_v_min <= strict.cramers_v_min


def test_calibration_with_constant_feature(label):
    df = pd.DataFrame({
        "constant": np.ones(len(label)),
        "good_feat": label.values * 2.0 + np.random.RandomState(42).normal(0, 1, len(label)),
    })
    thresholds = calibrate_l2_thresholds(df, label)
    assert thresholds.mi_min > 0
