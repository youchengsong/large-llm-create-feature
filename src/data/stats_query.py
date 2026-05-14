"""按需统计查询：在沙盒中执行 LLM 的数据探索代码，返回文本摘要。"""

import multiprocessing
import os
import tempfile
import traceback

import pandas as pd
import numpy as np

from src.evaluation.sandbox import check_code_safety, _ensure_data_cache

MAX_QUERY_TIMEOUT = 15
MAX_RESULT_CHARS = 2000


def _run_query_in_process(code, data_dir, result_path, error_queue):
    try:
        import builtins
        import pandas as pd
        import numpy as np

        open_df = pd.read_parquet(os.path.join(data_dir, "open.parquet"))
        tx_df = pd.read_parquet(os.path.join(data_dir, "tx.parquet"))
        close_df = pd.read_parquet(os.path.join(data_dir, "close.parquet"))

        safe_builtins = {k: v for k, v in vars(builtins).items()
                         if k not in {"open", "exec", "eval", "compile", "__import__",
                                      "globals", "locals", "getattr", "setattr", "delattr",
                                      "breakpoint", "exit", "quit"}}

        from src.evaluation.sandbox import _safe_import
        safe_builtins["__import__"] = _safe_import

        global_ns = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "np": np,
            "open_df": open_df,
            "tx_df": tx_df,
            "close_df": close_df,
        }
        exec(code, global_ns, global_ns)

        if "query_result" not in global_ns:
            error_queue.put("代码未定义 'query_result' 变量")
            return

        qr = global_ns["query_result"]
        if isinstance(qr, pd.DataFrame):
            text = qr.to_string(max_rows=30, max_cols=10)
        elif isinstance(qr, pd.Series):
            text = qr.to_string(max_rows=30)
        else:
            text = str(qr)

        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "\n... (截断)"

        with open(result_path, "w", encoding="utf-8") as f:
            f.write(text)
        error_queue.put(None)
    except Exception:
        error_queue.put(traceback.format_exc())


def execute_stats_query(
    code: str,
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
) -> tuple[str | None, str | None]:
    """在沙盒中执行统计查询代码，返回 (result_text, error_msg)。"""
    safe, reason = check_code_safety(code)
    if not safe:
        return None, f"安全检查失败: {reason}"

    data_dir = _ensure_data_cache(open_df, tx_df, close_df)
    result_path = tempfile.mktemp(suffix=".txt", prefix="query_result_")
    error_queue = multiprocessing.Queue()

    proc = multiprocessing.Process(
        target=_run_query_in_process,
        args=(code, data_dir, result_path, error_queue),
    )
    proc.start()
    proc.join(timeout=MAX_QUERY_TIMEOUT)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return None, f"查询超时 ({MAX_QUERY_TIMEOUT}s)"

    if error_queue.empty():
        return None, "查询异常: 子进程未返回结果"

    error = error_queue.get()
    if error:
        return None, error

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result_text = f.read()
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)

    return result_text, None
