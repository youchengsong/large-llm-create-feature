import json

import numpy as np
import pandas as pd
import pytest

from knowfeat.data.loader import Stage1Data, Stage2Data, subset_stage1, subset_stage2
from knowfeat.data.split import DataSplit, get_or_make_split, make_split


def test_make_split_is_complete_and_reproducible():
    first = make_split(50, seed=7, test_size=0.2)
    second = make_split(50, seed=7, test_size=0.2)

    assert np.array_equal(first.dev_idx, second.dev_idx)
    assert np.array_equal(first.test_idx, second.test_idx)
    assert np.array_equal(
        np.sort(np.concatenate([first.dev_idx, first.test_idx])), np.arange(50)
    )


@pytest.mark.parametrize(
    "dev_idx,test_idx",
    [([0, 0], [1, 2]), ([0, 1], [1, 2]), ([0, 3], [1, 2])],
)
def test_invalid_split_is_rejected(dev_idx, test_idx):
    with pytest.raises(ValueError):
        DataSplit(dev_idx, test_idx, n_total=3)


def test_cache_mismatch_is_regenerated(tmp_path):
    cache = tmp_path / "split.json"
    original = get_or_make_split(40, cache_path=cache, seed=1, test_size=0.2)
    refreshed = get_or_make_split(40, cache_path=cache, seed=2, test_size=0.25)

    assert original.seed == 1
    assert refreshed.seed == 2
    assert refreshed.test_size == 0.25
    assert len(refreshed.test_idx) == 10


def test_invalid_cache_is_regenerated(tmp_path):
    cache = tmp_path / "split.json"
    cache.write_text(json.dumps({"n_total": 4, "dev_idx": [0, 0], "test_idx": [1, 2]}))

    split = get_or_make_split(4, cache_path=cache)

    assert np.array_equal(
        np.sort(np.concatenate([split.dev_idx, split.test_idx])), np.arange(4)
    )


def test_subset_stage1_preserves_selected_order():
    data = Stage1Data(
        df=pd.DataFrame({"row_id": [10, 11, 12], "x": [1, 2, 3], "y": [0, 1, 0]}),
        baseline_features=pd.DataFrame({"x": [1, 2, 3]}),
        label_col="y",
        id_col="row_id",
        feature_cols=["x"],
    )

    subset = subset_stage1(data, np.array([2, 0]))

    assert subset.df["row_id"].tolist() == [12, 10]
    assert subset.baseline_features["x"].tolist() == [3, 1]
    assert subset.feature_cols == ["x"]


def test_subset_stage2_aligns_transactions_by_id():
    data = Stage2Data(
        open_df=pd.DataFrame({"wallet_id": [1]}),
        tx_df=pd.DataFrame({"tx_id": [30, 10, 20], "amount": [3, 1, 2]}),
        close_df=pd.DataFrame({"wallet_id": [1]}),
        baseline_features=pd.DataFrame({
            "tx_id_fe": [10, 20, 30],
            "is_risk": [0, 1, 0],
        }),
    )

    subset = subset_stage2(data, np.array([2, 0]))

    assert subset.baseline_features["tx_id_fe"].tolist() == [30, 10]
    assert subset.tx_df["tx_id"].tolist() == [30, 10]
    assert subset.open_df is data.open_df
    assert subset.close_df is data.close_df


def test_subset_stage2_rejects_missing_transaction_ids():
    data = Stage2Data(
        open_df=pd.DataFrame(),
        tx_df=pd.DataFrame({"tx_id": [10]}),
        close_df=pd.DataFrame(),
        baseline_features=pd.DataFrame({"tx_id_fe": [99], "is_risk": [1]}),
    )

    with pytest.raises(ValueError, match="absent from tx_df"):
        subset_stage2(data, np.array([0]))
