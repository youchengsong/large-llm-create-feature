"""特征执行沙盒：安全检查 + 超时执行 + 输出验证。"""

import ast
import multiprocessing
import os
import tempfile
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

    has_result_assignment = False
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

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "result":
                    has_result_assignment = True

    if not has_result_assignment:
        return False, "代码未包含 'result = ...' 赋值语句"

    return True, ""


def _safe_import(name, *args, **kwargs):
    """只允许白名单模块的 import。"""
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"禁止导入模块: {name}")
    import builtins
    return builtins.__import__(name, *args, **kwargs)


def _run_in_process(code, data_dir, result_path, error_queue):
    """在子进程中执行特征代码。"""
    try:
        import builtins
        import pandas as pd
        import numpy as np
        open_df = pd.read_parquet(os.path.join(data_dir, "open.parquet"))
        tx_df = pd.read_parquet(os.path.join(data_dir, "tx.parquet"))
        close_df = pd.read_parquet(os.path.join(data_dir, "close.parquet"))

        safe_builtins = {k: v for k, v in vars(builtins).items()
                         if k not in _FORBIDDEN_BUILTINS}
        safe_builtins["__import__"] = _safe_import

        global_ns = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "np": np,
            "open_df": open_df,
            "tx_df": tx_df,
            "close_df": close_df,
        }
        local_ns = global_ns
        exec(code, global_ns, local_ns)

        if "result" not in local_ns:
            error_queue.put("代码未定义 'result' 变量，请将特征 DataFrame 赋值给 result")
            return

        result = local_ns["result"]
        if not isinstance(result, pd.DataFrame):
            error_queue.put(f"result 类型应为 DataFrame，实际为 {type(result).__name__}")
            return

        result.to_parquet(result_path)
        error_queue.put(None)
    except Exception:
        error_queue.put(traceback.format_exc())


_cached_data_dir: str | None = None
_cached_data_id: tuple[int, int, int] | None = None


def _ensure_data_cache(open_df: pd.DataFrame, tx_df: pd.DataFrame, close_df: pd.DataFrame) -> str:
    """将 DataFrame 缓存为 Parquet 文件，避免重复序列化。"""
    global _cached_data_dir, _cached_data_id
    data_id = (id(open_df), id(tx_df), id(close_df))
    if _cached_data_dir and os.path.isdir(_cached_data_dir) and _cached_data_id == data_id:
        return _cached_data_dir

    data_dir = tempfile.mkdtemp(prefix="sandbox_data_")
    open_df.to_parquet(os.path.join(data_dir, "open.parquet"))
    tx_df.to_parquet(os.path.join(data_dir, "tx.parquet"))
    close_df.to_parquet(os.path.join(data_dir, "close.parquet"))
    _cached_data_dir = data_dir
    _cached_data_id = data_id
    return data_dir


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

    data_dir = _ensure_data_cache(open_df, tx_df, close_df)
    result_path = tempfile.mktemp(suffix=".parquet", prefix="sandbox_result_")
    error_queue = multiprocessing.Queue()

    proc = multiprocessing.Process(
        target=_run_in_process,
        args=(code, data_dir, result_path, error_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return None, f"执行超时 ({timeout}s)"

    if error_queue.empty():
        return None, "执行异常: 子进程未返回结果"

    error = error_queue.get()
    if error:
        return None, error

    try:
        result_df = pd.read_parquet(result_path)
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)
    return result_df, None


def validate_output(feature_df: pd.DataFrame) -> tuple[bool, str]:
    """检查特征 DataFrame 输出格式和质量。"""
    if feature_df is None or feature_df.empty:
        return False, "输出为空"

    has_wallet_id = "wallet_id" in feature_df.columns
    has_tx_id = "tx_id" in feature_df.columns
    if not has_wallet_id and not has_tx_id:
        return False, "输出必须包含 wallet_id 列（账户级特征）"

    if has_tx_id and not has_wallet_id:
        return False, "输出使用 tx_id 格式，请改为 wallet_id 格式（按账户聚合）"

    id_col = "wallet_id"
    feature_cols = [c for c in feature_df.columns if c != id_col]
    if not feature_cols:
        return False, "输出不包含任何特征列"

    if feature_df[id_col].duplicated().any():
        return False, f"wallet_id 存在重复值（{feature_df[id_col].duplicated().sum()} 行），请确保每个 wallet_id 只有一行"

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
