"""下载 6 个公开数据集到 data/public/ 目录。

OpenML: credit-g(31), diabetes(37), blood-transfusion(1464),
        bank-marketing(1461), breast-w(15)
Kaggle: heart (从 fedesoriano/heart-failure-prediction 下载)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "public"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPENML_DATASETS = {
    "credit_g": 31,
    "diabetes": 37,
    "blood_transfusion": 1464,
    "bank_marketing": 1461,
    "breast_w": 15,
    "adult": 1590,
}

DATASET_TARGET_MAP = {
    "credit_g": "class",
    "diabetes": "class",
    "blood_transfusion": "Class",
    "bank_marketing": "y",
    "breast_w": "Class",
    "heart": "HeartDisease",
    "adult": "class",
}


def download_openml(name: str, dataset_id: int, output_dir: Path) -> Path:
    """从 OpenML 下载数据集并保存为 CSV。"""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        logger.error("需要 scikit-learn: pip install scikit-learn")
        sys.exit(1)

    output_path = output_dir / f"{name}.csv"
    if output_path.exists():
        logger.info("已存在，跳过: %s", output_path)
        return output_path

    logger.info("从 OpenML 下载 %s (id=%d)...", name, dataset_id)
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")

    df = data.data.copy()
    target_name = DATASET_TARGET_MAP.get(name, "target")

    if data.target is not None:
        df[target_name] = data.target

    if name == "breast_w":
        if target_name in df.columns:
            df[target_name] = df[target_name].map(
                {"benign": 0, "malignant": 1, 2: 0, 4: 1}
            )
            orig_cols = df.columns.tolist()
            clean_cols = [
                c.replace("-", "_").replace(" ", "_") for c in orig_cols
            ]
            df.columns = clean_cols

    if name == "diabetes":
        if target_name in df.columns:
            df[target_name] = df[target_name].map(
                {"tested_positive": 1, "tested_negative": 0}
            ).fillna(df[target_name])

    if name == "credit_g":
        if target_name in df.columns:
            df[target_name] = df[target_name].map(
                {"good": 0, "bad": 1}
            ).fillna(df[target_name])

    if name == "blood_transfusion":
        if target_name in df.columns:
            df[target_name] = df[target_name].astype(int)

    if name == "bank_marketing":
        if target_name in df.columns:
            val_map = {"yes": 1, "no": 0, "1": 1, "2": 0}
            df[target_name] = df[target_name].map(val_map).fillna(df[target_name])

    if name == "adult":
        if target_name in df.columns:
            val_map = {"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1}
            df[target_name] = df[target_name].map(val_map).fillna(df[target_name])

    df.to_csv(output_path, index=False)
    logger.info("保存: %s (%d rows, %d cols)", output_path, *df.shape)
    return output_path


def download_heart(output_dir: Path) -> Path:
    """下载 Heart Disease 数据集。

    先尝试从 Kaggle 下载，失败则从 UCI ML Repository 下载。
    """
    output_path = output_dir / "heart.csv"
    if output_path.exists():
        logger.info("已存在，跳过: %s", output_path)
        return output_path

    # 尝试从 Kaggle API 下载
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d",
             "fedesoriano/heart-failure-prediction",
             "-p", str(output_dir), "--unzip"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and output_path.exists():
            logger.info("从 Kaggle 下载成功: %s", output_path)
            return output_path
        logger.warning("Kaggle 下载失败: %s", result.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("Kaggle CLI 不可用，尝试从 UCI 下载")

    # Fallback: 从多个来源组合 (Cleveland + Hungarian + Switzerland + VA Long Beach)
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
    ]

    col_names = [
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
        "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
        "Oldpeak", "ST_Slope", "ca", "thal", "HeartDisease",
    ]

    frames = []
    for url in urls:
        try:
            df = pd.read_csv(url, header=None, names=col_names, na_values="?")
            frames.append(df)
            logger.info("下载: %s (%d rows)", url.split("/")[-1], len(df))
        except Exception as e:
            logger.warning("下载失败 %s: %s", url, e)

    if not frames:
        logger.error("无法下载 Heart Disease 数据集，请手动下载到 %s", output_path)
        return output_path

    df = pd.concat(frames, ignore_index=True)
    df["HeartDisease"] = (df["HeartDisease"] > 0).astype(int)
    df = df.drop(columns=["ca", "thal"], errors="ignore")
    df.to_csv(output_path, index=False)
    logger.info("保存: %s (%d rows, %d cols)", output_path, *df.shape)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="下载公开数据集")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="指定数据集名称，默认下载全部")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = args.datasets or list(OPENML_DATASETS.keys()) + ["heart"]

    for name in datasets_to_download:
        if name == "heart":
            download_heart(args.output_dir)
        elif name in OPENML_DATASETS:
            download_openml(name, OPENML_DATASETS[name], args.output_dir)
        else:
            logger.warning("未知数据集: %s", name)

    logger.info("下载完成。数据目录: %s", args.output_dir)


if __name__ == "__main__":
    main()
