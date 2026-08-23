import json

import numpy as np
import pytest

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
