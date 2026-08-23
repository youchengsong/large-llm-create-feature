import pandas as pd

from scripts.evaluate_reduced_baseline import prepare_numeric


def test_prepare_numeric_excludes_identifiers_and_labels():
    frame = pd.DataFrame({
        "row_id": [10, 11],
        "class": [0, 1],
        "numeric": [1.0, None],
        "category": ["a", "b"],
    })

    prepared = prepare_numeric(frame)

    assert list(prepared.columns) == ["numeric", "category"]
    assert prepared["numeric"].tolist() == [1.0, -999.0]
    assert prepared["category"].tolist() == [0, 1]
