"""tests/test_sandbox.py — 测试特征执行沙盒的安全检查、超时、输出格式验证。"""

import pandas as pd
import pytest

from src.evaluation.sandbox import (
    ALLOWED_MODULES,
    check_code_safety,
    execute_feature_code,
    validate_output,
)


# ---------- 测试数据 ----------

@pytest.fixture
def sample_dfs():
    open_df = pd.DataFrame({
        "wallet_id": ["w1", "w2", "w3"],
        "balance": [100.0, 200.0, 300.0],
    })
    tx_df = pd.DataFrame({
        "tx_id": ["t1", "t2", "t3"],
        "src_wallet": ["w1", "w2", "w3"],
        "dst_wallet": ["w2", "w3", "w1"],
        "amount": [10.0, 20.0, 30.0],
        "is_risk": [0, 1, 0],
    })
    close_df = pd.DataFrame({
        "wallet_id": ["w1"],
        "close_date": ["2025-01-01"],
    })
    return open_df, tx_df, close_df


# ========== check_code_safety ==========

class TestCheckCodeSafety:
    def test_safe_code(self):
        code = "import pandas as pd\nimport numpy as np\nresult = pd.DataFrame({'a': [1]})"
        safe, reason = check_code_safety(code)
        assert safe is True
        assert reason == ""

    def test_allowed_modules(self):
        for mod in ALLOWED_MODULES:
            safe, _ = check_code_safety(f"import {mod}")
            assert safe is True

    def test_forbidden_import_os(self):
        safe, reason = check_code_safety("import os")
        assert safe is False
        assert "os" in reason

    def test_forbidden_import_subprocess(self):
        safe, reason = check_code_safety("import subprocess")
        assert safe is False

    def test_forbidden_from_import(self):
        safe, reason = check_code_safety("from os.path import join")
        assert safe is False

    def test_forbidden_requests(self):
        safe, reason = check_code_safety("import requests")
        assert safe is False

    def test_forbidden_builtin_open(self):
        safe, reason = check_code_safety("f = open('test.txt')")
        assert safe is False
        assert "open" in reason

    def test_forbidden_builtin_exec(self):
        safe, reason = check_code_safety("exec('print(1)')")
        assert safe is False

    def test_forbidden_builtin_eval(self):
        safe, reason = check_code_safety("x = eval('1+1')")
        assert safe is False

    def test_syntax_error(self):
        safe, reason = check_code_safety("def foo(:")
        assert safe is False
        assert "语法错误" in reason

    def test_forbidden_reference_sys(self):
        safe, reason = check_code_safety("x = sys.argv")
        assert safe is False

    def test_scipy_allowed(self):
        safe, _ = check_code_safety("from scipy.stats import entropy")
        assert safe is True

    def test_sklearn_allowed(self):
        safe, _ = check_code_safety("from sklearn.preprocessing import StandardScaler")
        assert safe is True


# ========== execute_feature_code ==========

class TestExecuteFeatureCode:
    def test_valid_code(self, sample_dfs):
        open_df, tx_df, close_df = sample_dfs
        code = """
import pandas as pd
result = pd.DataFrame({'tx_id': tx_df['tx_id'], 'feat': tx_df['amount'] * 2})
"""
        df, err = execute_feature_code(code, open_df, tx_df, close_df, timeout=10)
        assert err is None
        assert df is not None
        assert "feat" in df.columns
        assert len(df) == 3

    def test_unsafe_code_rejected(self, sample_dfs):
        open_df, tx_df, close_df = sample_dfs
        code = "import os\nresult = pd.DataFrame()"
        df, err = execute_feature_code(code, open_df, tx_df, close_df, timeout=10)
        assert df is None
        assert "安全检查失败" in err

    def test_no_result_variable(self, sample_dfs):
        open_df, tx_df, close_df = sample_dfs
        code = "x = 42"
        df, err = execute_feature_code(code, open_df, tx_df, close_df, timeout=10)
        assert df is None
        assert "result" in err

    def test_result_not_dataframe(self, sample_dfs):
        open_df, tx_df, close_df = sample_dfs
        code = "result = [1, 2, 3]"
        df, err = execute_feature_code(code, open_df, tx_df, close_df, timeout=10)
        assert df is None
        assert "DataFrame" in err

    def test_timeout(self, sample_dfs):
        open_df, tx_df, close_df = sample_dfs
        code = """
import time
time.sleep(60)
result = pd.DataFrame()
"""
        # time is not in ALLOWED_MODULES, so this should fail at safety check
        # Use a loop instead to test actual timeout
        code_loop = """
import pandas as pd
while True:
    pass
result = pd.DataFrame()
"""
        # This code is safe (only imports pandas) but will hang
        df, err = execute_feature_code(code_loop, open_df, tx_df, close_df, timeout=2)
        assert df is None
        assert "超时" in err

    def test_runtime_error(self, sample_dfs):
        open_df, tx_df, close_df = sample_dfs
        code = "result = 1 / 0"
        df, err = execute_feature_code(code, open_df, tx_df, close_df, timeout=10)
        assert df is None
        assert err is not None

    def test_uses_input_dfs(self, sample_dfs):
        open_df, tx_df, close_df = sample_dfs
        code = """
import pandas as pd
result = pd.DataFrame({
    'wallet_id': open_df['wallet_id'],
    'bal_double': open_df['balance'] * 2,
})
"""
        df, err = execute_feature_code(code, open_df, tx_df, close_df, timeout=10)
        assert err is None
        assert list(df['bal_double']) == [200.0, 400.0, 600.0]


# ========== validate_output ==========

class TestValidateOutput:
    def test_valid_output_tx_id(self):
        df = pd.DataFrame({"tx_id": [1, 2, 3], "feat": [0.1, 0.5, 0.9]})
        ok, reason = validate_output(df)
        assert ok is True

    def test_valid_output_wallet_id(self):
        df = pd.DataFrame({"wallet_id": ["w1", "w2"], "feat": [1, 0]})
        ok, reason = validate_output(df)
        assert ok is True

    def test_empty_df(self):
        ok, reason = validate_output(pd.DataFrame())
        assert ok is False
        assert "空" in reason

    def test_none_input(self):
        ok, reason = validate_output(None)
        assert ok is False

    def test_no_id_column(self):
        df = pd.DataFrame({"feat": [1, 2, 3]})
        ok, reason = validate_output(df)
        assert ok is False
        assert "tx_id" in reason or "wallet_id" in reason

    def test_no_feature_columns(self):
        df = pd.DataFrame({"tx_id": [1, 2, 3]})
        ok, reason = validate_output(df)
        assert ok is False
        assert "特征" in reason

    def test_high_nan_ratio(self):
        df = pd.DataFrame({
            "tx_id": range(10),
            "feat": [None]*6 + [1.0]*4,
        })
        ok, reason = validate_output(df)
        assert ok is False
        assert "NaN" in reason

    def test_all_nan(self):
        df = pd.DataFrame({"tx_id": [1, 2], "feat": [None, None]})
        ok, reason = validate_output(df)
        assert ok is False

    def test_constant_feature(self):
        df = pd.DataFrame({"tx_id": [1, 2, 3], "feat": [0, 0, 0]})
        ok, reason = validate_output(df)
        assert ok is False
        assert "唯一值" in reason
