"""Unified data split for no-leakage evaluation (single source of truth).

Design goal (addressing feature-selection leakage raised by ICDM reviewers):

  D  --stratified, fixed seed-->  D_dev (default 80%)  +  D_test (default 20%, frozen)

Rules:
  * Feature **generation** (explore_data / schema statistics) may only see D_dev.
  * Feature **selection** (L2 noise calibration / L3 evaluate_feature / forward selection)
    may only use D_dev; L3's internal 5-seed CV splits within D_dev, D_test never
    participates in any "accept/reject" decisions.
  * Final metrics (Table III / V) are computed once on D_test with bootstrap resampling
    to report mean +/- variance.

A feature is fundamentally "a deterministic Python transformation code"; the code itself
contains no labels. The root cause of leakage is "which data's labels are used to decide
accept/reject". Therefore this module only provides a stable, reproducible, disk-cacheable
(dev_idx, test_idx) split, shared between the KnowFeat generation side and the final
evaluation side, ensuring both use the same frozen test set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Default split parameters: globally fixed, no script should set them individually to ensure dev/test consistency.
DEFAULT_SPLIT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


@dataclass
class DataSplit:
    """A frozen dev/test split (based on position indices 0..n-1)."""

    dev_idx: np.ndarray
    test_idx: np.ndarray
    seed: int = DEFAULT_SPLIT_SEED
    test_size: float = DEFAULT_TEST_SIZE
    n_total: int = 0

    def __post_init__(self):
        self.dev_idx = np.asarray(self.dev_idx, dtype=int)
        self.test_idx = np.asarray(self.test_idx, dtype=int)
        if self.n_total == 0:
            self.n_total = len(self.dev_idx) + len(self.test_idx)
        if self.n_total < 0:
            raise ValueError("n_total must be non-negative")
        if self.dev_idx.ndim != 1 or self.test_idx.ndim != 1:
            raise ValueError("dev_idx and test_idx must be one-dimensional")
        if len(np.unique(self.dev_idx)) != len(self.dev_idx):
            raise ValueError("dev_idx contains duplicate positions")
        if len(np.unique(self.test_idx)) != len(self.test_idx):
            raise ValueError("test_idx contains duplicate positions")
        overlap = np.intersect1d(self.dev_idx, self.test_idx)
        if overlap.size > 0:
            raise ValueError(
                f"dev/test indices overlap by {overlap.size}, split is invalid (leakage risk)."
            )
        all_idx = np.concatenate([self.dev_idx, self.test_idx])
        if all_idx.size != self.n_total:
            raise ValueError(
                f"dev/test sizes ({all_idx.size}) do not match n_total={self.n_total}"
            )
        if all_idx.size and (all_idx.min() < 0 or all_idx.max() >= self.n_total):
            raise ValueError("dev/test indices are outside the dataset bounds")
        if not np.array_equal(np.sort(all_idx), np.arange(self.n_total)):
            raise ValueError("dev/test indices do not form a complete partition")
        if not 0 < float(self.test_size) < 1:
            raise ValueError("test_size must be between 0 and 1")

    # -- Convenience slicing -------------------------------------------------
    def dev(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.iloc[self.dev_idx].reset_index(drop=True)

    def test(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.iloc[self.test_idx].reset_index(drop=True)

    def dev_labels(self, y: pd.Series) -> pd.Series:
        return y.iloc[self.dev_idx].reset_index(drop=True)

    def test_labels(self, y: pd.Series) -> pd.Series:
        return y.iloc[self.test_idx].reset_index(drop=True)

    # -- Persistence (cross-script / cross-process reuse of the same frozen split) --
    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": int(self.seed),
                    "test_size": float(self.test_size),
                    "n_total": int(self.n_total),
                    "dev_idx": self.dev_idx.tolist(),
                    "test_idx": self.test_idx.tolist(),
                },
                f,
            )
        logger.info(
            "Split saved -> %s (dev=%d, test=%d)",
            path, len(self.dev_idx), len(self.test_idx),
        )
        return path

    @classmethod
    def from_json(cls, path: Path) -> "DataSplit":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            dev_idx=np.asarray(d["dev_idx"], dtype=int),
            test_idx=np.asarray(d["test_idx"], dtype=int),
            seed=int(d.get("seed", DEFAULT_SPLIT_SEED)),
            test_size=float(d.get("test_size", DEFAULT_TEST_SIZE)),
            n_total=int(d.get("n_total", 0)),
        )


def make_split(
    n_samples: int,
    labels: Optional[pd.Series] = None,
    seed: int = DEFAULT_SPLIT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> DataSplit:
    """Generate a reproducible frozen dev/test split.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    labels : pd.Series, optional
        If provided, perform stratified splitting (strongly recommended, especially for
        highly imbalanced AML scenarios).
    seed : int
        Split random seed (globally fixed, do not confuse with model evaluation seeds).
    test_size : float
        Frozen test set proportion.

    Returns
    -------
    DataSplit
    """
    idx = np.arange(n_samples)
    stratify = None
    if labels is not None:
        y = np.asarray(labels)
        if len(y) != n_samples:
            raise ValueError(
                f"labels length {len(y)} != n_samples {n_samples}"
            )
        # Only stratify when each class has at least 2 samples
        _, counts = np.unique(y, return_counts=True)
        if counts.min() >= 2:
            stratify = y

    dev_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=stratify,
    )
    dev_idx.sort()
    test_idx.sort()
    split = DataSplit(
        dev_idx=dev_idx, test_idx=test_idx,
        seed=seed, test_size=test_size, n_total=n_samples,
    )
    logger.info(
        "Generated frozen split: n=%d -> dev=%d (%.1f%%), test=%d (%.1f%%), seed=%d, stratified=%s",
        n_samples, len(dev_idx), 100 * len(dev_idx) / n_samples,
        len(test_idx), 100 * len(test_idx) / n_samples,
        seed, stratify is not None,
    )
    return split


def get_or_make_split(
    n_samples: int,
    labels: Optional[pd.Series] = None,
    cache_path: Optional[Path] = None,
    seed: int = DEFAULT_SPLIT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> DataSplit:
    """Reuse cached split if it exists and matches the size, otherwise create and save a new one.

    This ensures the KnowFeat generation phase and the final evaluation phase
    (which may be different processes/scripts) use **exactly the same** frozen test set.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                split = DataSplit.from_json(cache_path)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                logger.warning("Invalid cached split (%s), regenerating: %s", exc, cache_path)
            else:
                same_seed = split.seed == seed
                same_test_size = np.isclose(split.test_size, test_size)
                if split.n_total == n_samples and same_seed and same_test_size:
                    logger.info("Reusing cached split: %s", cache_path)
                    return split
                logger.warning(
                    "Cached split mismatch (n=%d/%d, seed=%d/%d, test_size=%s/%s), regenerating: %s",
                    split.n_total, n_samples, split.seed, seed,
                    split.test_size, test_size, cache_path,
                )
    split = make_split(n_samples, labels=labels, seed=seed, test_size=test_size)
    if cache_path is not None:
        split.to_json(cache_path)
    return split
