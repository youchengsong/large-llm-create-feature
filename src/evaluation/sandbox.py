"""特征执行沙盒：安全检查 + 超时执行 + 输出验证。"""

import ast
import multiprocessing
import traceback

import numpy as np
import pandas as pd

ALLOWED_MODULES = {"pandas", "numpy", "scipy", "sklearn"}

_FORBIDDEN_NAMES = {"os", "sys", "subprocess", "shutil", "pathlib", "socket",
                    "http", "urllib", "requests", "ftplib", "smtplib",
                    "ctypes", "importlib", "pickle", "shelve", "signal"}

_FORBIDDEN_BUILTINS = {"open", "exec", "eval", "compile", "__import__",
                       "globals", "locals", "getattr", "setattr", "delattr",
                       "breakpoint", "exit", "quit"}


def check_code_safety(code: str) -> tuple[bool, str]:
    """AST 解析检查代码安全性。返回 (is_safe, reason)。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"语法错误: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return False, f"禁止导入模块: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return False, f"禁止导入模块: {node.module}"

        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _FORBIDDEN_BUILTINS:
                return False, f"禁止调用: {name}()"

        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES and not isinstance(node.ctx, ast.Store):
                return False, f"禁止引用: {node.id}"

    return True, ""


def _safe_import(name, *args, **kwargs):
    """只允许白名单模块的 import。"""
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"禁止导入模块: {name}")
    import builtins
    return builtins.__import__(name, *args, **kwargs)


def _run_in_process(code, open_dict, tx_dict, close_dict, result_queue):
    """在子进程中执行特征代码。"""
    try:
        import builtins
        import pandas as pd
        import numpy as np
        open_df = pd.DataFrame(open_dict)
        tx_df = pd.DataFrame(tx_dict)
        close_df = pd.DataFrame(close_dict)

        safe_builtins = {k: v for k, v in vars(builtins).items()
                         if k not in _FORBIDDEN_BUILTINS}
        safe_builtins["__import__"] = _safe_import

        local_ns = {
            "pd": pd,
            "np": np,
            "open_df": open_df,
            "tx_df": tx_df,
            "close_df": close_df,
        }
        exec(code, {"__builtins__": safe_builtins}, local_ns)

        if "result" not in local_ns:
            result_queue.put((None, "代码未定义 'result' 变量，请将特征 DataFrame 赋值给 result"))
            return

        result = local_ns["result"]
        if not isinstance(result, pd.DataFrame):
            result_queue.put((None, f"result 类型应为 DataFrame，实际为 {type(result).__name__}"))
            return

        result_queue.put((result.to_dict(), None))
    except Exception:
        result_queue.put((None, traceback.format_exc()))


def execute_feature_code(
    code: str,
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
    timeout: int = 30,
) -> tuple[pd.DataFrame | None, str | None]:
    """沙盒执行特征代码。返回 (feature_df, error_msg)。"""
    safe, reason = check_code_safety(code)
    if not safe:
        return None, f"安全检查失败: {reason}"

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_in_process,
        args=(code, open_df.to_dict(), tx_df.to_dict(), close_df.to_dict(), result_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return None, f"执行超时 ({timeout}s)"

    if result_queue.empty():
        return None, "执行异常: 子进程未返回结果"

    data, error = result_queue.get()
    if error:
        return None, error
    return pd.DataFrame(data), None


def validate_output(feature_df: pd.DataFrame) -> tuple[bool, str]:
    """检查特征 DataFrame 输出格式和质量。"""
    if feature_df is None or feature_df.empty:
        return False, "输出为空"

    has_id = "tx_id" in feature_df.columns or "wallet_id" in feature_df.columns
    if not has_id:
        return False, "输出必须包含 tx_id 或 wallet_id 列"

    id_col = "tx_id" if "tx_id" in feature_df.columns else "wallet_id"
    feature_cols = [c for c in feature_df.columns if c != id_col]
    if not feature_cols:
        return False, "输出不包含任何特征列"

    for col in feature_cols:
        series = feature_df[col]

        nan_ratio = series.isna().mean()
        if nan_ratio > 0.5:
            return False, f"特征 {col} NaN 比例过高: {nan_ratio:.1%}"

        non_null = series.dropna()
        if len(non_null) == 0:
            return False, f"特征 {col} 全为 NaN"

        if non_null.nunique() <= 1:
            return False, f"特征 {col} 只有一个唯一值 (全0/全1)"

    return True, ""
