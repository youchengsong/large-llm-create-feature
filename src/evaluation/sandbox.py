"""特征执行沙盒：支持本地进程隔离 (local) 和 E2B 云端 microVM (e2b) 两种模式。

切换方式（任选其一）：
  1. 环境变量: SANDBOX_MODE=e2b（默认 local）
  2. 代码: sandbox.set_mode("e2b")

使用 E2B 模式需要:
  - pip install e2b-code-interpreter
  - 设置环境变量 E2B_API_KEY
"""

import ast
import io
import logging
import multiprocessing
import os
import tempfile
import traceback

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 沙盒模式配置
# ============================================================

_sandbox_mode: str = os.environ.get("SANDBOX_MODE", "local")


def set_mode(mode: str) -> None:
    """设置沙盒模式: 'local' 或 'e2b'。"""
    global _sandbox_mode
    if mode not in ("local", "e2b"):
        raise ValueError(f"不支持的沙盒模式: {mode}，可选: local, e2b")
    _sandbox_mode = mode
    logger.info("沙盒模式切换为: %s", mode)


def get_mode() -> str:
    """获取当前沙盒模式。"""
    return _sandbox_mode


# ============================================================
# 通用：AST 安全检查 + 输出验证（两种模式共用）
# ============================================================

ALLOWED_MODULES = {"pandas", "numpy", "scipy", "sklearn", "datetime", "collections", "math"}

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


# ============================================================
# 统一入口
# ============================================================

def execute_feature_code(
    code: str,
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
    timeout: int = 90,
) -> tuple[pd.DataFrame | None, str | None]:
    """沙盒执行特征代码。根据当前模式分发到 local 或 e2b 后端。

    返回 (feature_df, error_msg)。
    """
    safe, reason = check_code_safety(code)
    if not safe:
        return None, f"安全检查失败: {reason}"

    if _sandbox_mode == "e2b":
        return _execute_e2b(code, open_df, tx_df, close_df, timeout)
    else:
        return _execute_local(code, open_df, tx_df, close_df, timeout)


# ============================================================
# 模式 A：本地进程隔离（原有逻辑）
# ============================================================

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
            import inspect
            func_candidates = [
                v for k, v in local_ns.items()
                if callable(v) and not k.startswith("_")
                and k not in ("pd", "np") and inspect.isfunction(v)
            ]
            if func_candidates:
                func = func_candidates[0]
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                kwargs = {}
                for p in params:
                    if p in ("open_df", "tx_df", "close_df"):
                        kwargs[p] = local_ns[p]
                local_ns["result"] = func(**kwargs)
            else:
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


def _execute_local(
    code: str,
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
    timeout: int = 90,
) -> tuple[pd.DataFrame | None, str | None]:
    """本地进程隔离模式执行特征代码。"""
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


# ============================================================
# 模式 B：E2B 云端 microVM 沙盒
# ============================================================

_e2b_sandbox = None
_e2b_data_uploaded: bool = False
_e2b_data_id: tuple[int, int, int] | None = None


def _get_e2b_sandbox(timeout: int = 300):
    """获取或创建 E2B 沙盒实例（复用同一个沙盒，数据只上传一次）。"""
    global _e2b_sandbox
    if _e2b_sandbox is not None:
        return _e2b_sandbox

    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        raise ImportError(
            "E2B 模式需要安装 e2b-code-interpreter: pip install e2b-code-interpreter\n"
            "并设置环境变量 E2B_API_KEY"
        )

    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        raise ValueError("E2B 模式需要设置环境变量 E2B_API_KEY")

    _e2b_sandbox = Sandbox(timeout=timeout, api_key=api_key)
    _e2b_sandbox.run_code("import pandas as pd; import numpy as np")
    logger.info("E2B 沙盒已创建 (sandbox_id=%s)", getattr(_e2b_sandbox, 'sandbox_id', 'unknown'))
    return _e2b_sandbox


def _upload_data_to_e2b(
    sbx,
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
) -> None:
    """将三张数据表上传到 E2B 沙盒（只在数据变化时重新上传）。"""
    global _e2b_data_uploaded, _e2b_data_id
    data_id = (id(open_df), id(tx_df), id(close_df))
    if _e2b_data_uploaded and _e2b_data_id == data_id:
        return

    for name, df in [("open", open_df), ("tx", tx_df), ("close", close_df)]:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        sbx.files.write(f"/home/user/{name}.parquet", buf)

    sbx.run_code(
        "open_df = pd.read_parquet('/home/user/open.parquet')\n"
        "tx_df = pd.read_parquet('/home/user/tx.parquet')\n"
        "close_df = pd.read_parquet('/home/user/close.parquet')"
    )

    _e2b_data_uploaded = True
    _e2b_data_id = data_id
    logger.info("数据已上传到 E2B 沙盒 (open=%s, tx=%s, close=%s)",
                open_df.shape, tx_df.shape, close_df.shape)


def _execute_e2b(
    code: str,
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
    timeout: int = 90,
) -> tuple[pd.DataFrame | None, str | None]:
    """E2B 云端 microVM 模式执行特征代码。"""
    try:
        sbx = _get_e2b_sandbox(timeout=600)
        _upload_data_to_e2b(sbx, open_df, tx_df, close_df)
    except Exception as e:
        return None, f"E2B 沙盒初始化失败: {e}"

    wrapped_code = (
        f"{code}\n\n"
        "# 保存结果\n"
        "import inspect as _inspect\n"
        "if 'result' not in dir():\n"
        "    _funcs = [v for k, v in list(locals().items())\n"
        "             if callable(v) and not k.startswith('_')\n"
        "             and k not in ('pd', 'np') and _inspect.isfunction(v)]\n"
        "    if _funcs:\n"
        "        _f = _funcs[0]\n"
        "        _sig = _inspect.signature(_f)\n"
        "        _kw = {p: locals()[p] for p in _sig.parameters if p in ('open_df', 'tx_df', 'close_df')}\n"
        "        result = _f(**_kw)\n"
        "if 'result' in dir() and hasattr(result, 'to_parquet'):\n"
        "    result.to_parquet('/home/user/result.parquet')\n"
        "    print('__RESULT_SAVED__')\n"
        "else:\n"
        "    print('__NO_RESULT__')\n"
    )

    try:
        execution = sbx.run_code(wrapped_code, timeout=timeout)
    except Exception as e:
        return None, f"E2B 执行异常: {e}"

    if execution.error:
        return None, f"{execution.error.name}: {execution.error.value}"

    stdout = ""
    if hasattr(execution, 'logs'):
        logs = execution.logs
        if hasattr(logs, 'stdout'):
            stdout = "".join(msg.line if hasattr(msg, 'line') else str(msg) for msg in logs.stdout)
        elif isinstance(logs, str):
            stdout = logs

    if "__NO_RESULT__" in stdout:
        return None, "代码未定义 'result' 变量，请将特征 DataFrame 赋值给 result"

    if "__RESULT_SAVED__" not in stdout:
        return None, f"代码执行完成但未产出结果。输出: {stdout[:200]}"

    try:
        result_bytes = sbx.files.read("/home/user/result.parquet", format="bytes")
        result_df = pd.read_parquet(io.BytesIO(result_bytes))
    except Exception as e:
        return None, f"读取 E2B 结果失败: {e}"

    return result_df, None


def close_e2b_sandbox() -> None:
    """关闭 E2B 沙盒实例（实验结束时调用）。"""
    global _e2b_sandbox, _e2b_data_uploaded, _e2b_data_id
    if _e2b_sandbox is not None:
        try:
            _e2b_sandbox.kill()
            logger.info("E2B 沙盒已关闭")
        except Exception as e:
            logger.warning("关闭 E2B 沙盒失败: %s", e)
        _e2b_sandbox = None
        _e2b_data_uploaded = False
        _e2b_data_id = None
