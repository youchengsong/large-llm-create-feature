"""Tests for knowfeat.verification.sandbox."""

import pandas as pd
import pytest

from knowfeat.verification.sandbox import (
    SandboxResult,
    _extract_func_name,
    _validate_output,
    run_sandbox,
    run_sandbox_with_retry,
)


@pytest.fixture
def sample_dfs():
    open_df = pd.DataFrame({
        "wallet_id": [1, 2, 3],
        "wallet_type": [1, 2, 3],
    })
    tx_df = pd.DataFrame({
        "tx_id": [10, 20, 30],
        "src": [1, 2, 3],
        "dst": [2, 3, 1],
        "amount": [100.0, 200.0, 50.0],
        "is_risk": [0, 1, 0],
    })
    close_df = pd.DataFrame({
        "wallet_id": [3],
        "zs_timestamp": ["2025-01-01"],
    })
    return open_df, tx_df, close_df


GOOD_CODE = """\
import pandas as pd

def feature_tx_sum(open_df, tx_df, close_df):
    result = tx_df.groupby("src")["amount"].sum().reset_index()
    result.columns = ["wallet_id", "tx_sum"]
    return result
"""

CODE_NO_ID_COL = """\
import pandas as pd

def feature_bad(open_df, tx_df, close_df):
    return pd.DataFrame({"foo": [1, 2, 3]})
"""

CODE_EMPTY_DF = """\
import pandas as pd

def feature_empty(open_df, tx_df, close_df):
    return pd.DataFrame(columns=["wallet_id", "val"])
"""

CODE_RUNTIME_ERROR = """\
def feature_crash(open_df, tx_df, close_df):
    raise RuntimeError("boom")
"""

CODE_TIMEOUT = """\
import time

def feature_slow(open_df, tx_df, close_df):
    time.sleep(60)
    return open_df
"""

CODE_NO_FUNC = "x = 1 + 2"


class TestExtractFuncName:
    def test_normal(self):
        assert _extract_func_name(GOOD_CODE) == "feature_tx_sum"

    def test_no_func(self):
        with pytest.raises(ValueError, match="未找到函数定义"):
            _extract_func_name(CODE_NO_FUNC)


class TestValidateOutput:
    def test_valid_wallet_id(self):
        df = pd.DataFrame({"wallet_id": [1], "val": [0.5]})
        assert _validate_output(df) is None

    def test_valid_tx_id(self):
        df = pd.DataFrame({"tx_id": [1], "val": [0.5]})
        assert _validate_output(df) is None

    def test_missing_id_col(self):
        df = pd.DataFrame({"foo": [1]})
        err = _validate_output(df)
        assert err is not None
        assert "ID 列" in err

    def test_empty_df(self):
        df = pd.DataFrame(columns=["wallet_id", "val"])
        err = _validate_output(df)
        assert err is not None
        assert "空" in err

    def test_not_dataframe(self):
        err = _validate_output([1, 2, 3])
        assert err is not None
        assert "DataFrame" in err


class TestRunSandbox:
    def test_success(self, sample_dfs):
        open_df, tx_df, close_df = sample_dfs
        result = run_sandbox(GOOD_CODE, open_df, tx_df, close_df)
        assert result.success is True
        assert isinstance(result.df, pd.DataFrame)
        assert "wallet_id" in result.df.columns
        assert "tx_sum" in result.df.columns

    def test_no_id_col(self, sample_dfs):
        result = run_sandbox(CODE_NO_ID_COL, *sample_dfs)
        assert result.success is False
        assert result.error_type == "validation"
        assert "ID 列" in result.error

    def test_empty_output(self, sample_dfs):
        result = run_sandbox(CODE_EMPTY_DF, *sample_dfs)
        assert result.success is False
        assert result.error_type == "validation"

    def test_runtime_error(self, sample_dfs):
        result = run_sandbox(CODE_RUNTIME_ERROR, *sample_dfs)
        assert result.success is False
        assert result.error_type == "runtime"
        assert "boom" in result.error

    def test_timeout(self, sample_dfs):
        result = run_sandbox(CODE_TIMEOUT, *sample_dfs, timeout=2)
        assert result.success is False
        assert result.error_type == "timeout"

    def test_no_func_def(self, sample_dfs):
        result = run_sandbox(CODE_NO_FUNC, *sample_dfs)
        assert result.success is False
        assert result.error_type == "parse"


class TestRunSandboxWithRetry:
    def test_success(self, sample_dfs):
        result, history = run_sandbox_with_retry(
            GOOD_CODE, *sample_dfs
        )
        assert result.success is True
        assert history == []

    def test_failure(self, sample_dfs):
        result, history = run_sandbox_with_retry(
            CODE_RUNTIME_ERROR, *sample_dfs
        )
        assert result.success is False
