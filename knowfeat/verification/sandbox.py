"""L1 沙盒执行：隔离运行 Agent 生成的特征代码，验证输出格式。"""

import logging
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from knowfeat.config import KnowFeatConfig, VerificationConfig

logger = logging.getLogger(__name__)

REQUIRED_ID_COLS = {"wallet_id", "tx_id"}

_RUNNER_TEMPLATE = textwrap.dedent("""\
    import pickle, sys, pathlib

    data = pickle.loads(pathlib.Path(sys.argv[1]).read_bytes())
    open_df, tx_df, close_df = data["open_df"], data["tx_df"], data["close_df"]

    {code}

    result = {func_name}(open_df, tx_df, close_df)
    pathlib.Path(sys.argv[2]).write_bytes(pickle.dumps(result))
""")


@dataclass
class SandboxResult:
    """L1 沙盒执行结果。"""
    success: bool
    df: Optional[pd.DataFrame] = None
    error: str = ""
    error_type: str = ""


def _build_runner_script(code: str, func_name: str) -> str:
    return _RUNNER_TEMPLATE.format(code=code, func_name=func_name)


def _extract_func_name(code: str) -> str:
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and "(" in stripped:
            return stripped[4:stripped.index("(")]
    raise ValueError("代码中未找到函数定义（def ...）")


def _validate_output(df: pd.DataFrame) -> Optional[str]:
    if not isinstance(df, pd.DataFrame):
        return f"输出类型应为 DataFrame，实际为 {type(df).__name__}"
    if df.empty:
        return "输出 DataFrame 为空（0 行）"
    if not REQUIRED_ID_COLS & set(df.columns):
        return (
            f"输出 DataFrame 必须包含 {REQUIRED_ID_COLS} 中至少一个 ID 列，"
            f"实际列: {list(df.columns)}"
        )
    return None


def run_sandbox(
    code: str,
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
    timeout: int = 30,
) -> SandboxResult:
    """在子进程中隔离执行特征代码。

    Parameters
    ----------
    code : str
        Agent 生成的 Python 代码，必须包含一个函数定义
        ``def feature_xxx(open_df, tx_df, close_df) -> pd.DataFrame``。
    open_df, tx_df, close_df : pd.DataFrame
        Stage 2 三表数据。
    timeout : int
        子进程超时秒数，默认 30。

    Returns
    -------
    SandboxResult
        success=True 时 df 为输出 DataFrame；
        success=False 时 error 包含错误信息。
    """
    try:
        func_name = _extract_func_name(code)
    except ValueError as e:
        return SandboxResult(success=False, error=str(e), error_type="parse")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.pkl"
        output_path = tmpdir / "output.pkl"
        script_path = tmpdir / "runner.py"

        import pickle
        input_path.write_bytes(pickle.dumps({
            "open_df": open_df,
            "tx_df": tx_df,
            "close_df": close_df,
        }))

        script_path.write_text(
            _build_runner_script(code, func_name), encoding="utf-8"
        )

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path),
                 str(input_path), str(output_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                error=f"代码执行超时（>{timeout}s）",
                error_type="timeout",
            )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            return SandboxResult(
                success=False,
                error=f"代码执行失败:\n{stderr}",
                error_type="runtime",
            )

        try:
            result_df = pickle.loads(output_path.read_bytes())
        except Exception as e:
            return SandboxResult(
                success=False,
                error=f"无法加载输出结果: {e}",
                error_type="output",
            )

    validation_error = _validate_output(result_df)
    if validation_error:
        return SandboxResult(
            success=False,
            error=validation_error,
            error_type="validation",
        )

    logger.info(
        "L1 沙盒通过: %s, 输出 shape=%s", func_name, result_df.shape
    )
    return SandboxResult(success=True, df=result_df)


def run_sandbox_with_retry(
    code: str,
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
    cfg: Optional[KnowFeatConfig] = None,
    vcfg: Optional[VerificationConfig] = None,
) -> tuple[SandboxResult, list[SandboxResult]]:
    """执行沙盒，首次失败后返回结果供 Agent 修复重试。

    本函数本身只执行一次。调用方（react_agent）负责修复代码后
    再次调用本函数，最多重试 max_retries 次。

    Returns
    -------
    (result, history)
        result: 本次执行结果
        history: 空列表（保留接口，供调用方累积历史）
    """
    if vcfg is None:
        vcfg = cfg.verification if cfg is not None else VerificationConfig()

    result = run_sandbox(
        code=code,
        open_df=open_df,
        tx_df=tx_df,
        close_df=close_df,
        timeout=vcfg.sandbox_timeout_sec,
    )
    return result, []
