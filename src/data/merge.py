import pandas as pd


def merge_feature_to_transactions(
    feature_df: pd.DataFrame, tx_df: pd.DataFrame
) -> pd.DataFrame:
    result = tx_df.copy()

    if "tx_id" in feature_df.columns:
        return result.merge(feature_df, on="tx_id", how="left")

    if "wallet_id" in feature_df.columns:
        feat_cols = [c for c in feature_df.columns if c != "wallet_id"]

        src_rename = {"wallet_id": "src"}
        src_rename.update({c: f"{c}_src" for c in feat_cols})
        result = result.merge(
            feature_df.rename(columns=src_rename), on="src", how="left"
        )

        dst_rename = {"wallet_id": "dst"}
        dst_rename.update({c: f"{c}_dst" for c in feat_cols})
        result = result.merge(
            feature_df.rename(columns=dst_rename), on="dst", how="left"
        )

        return result

    raise ValueError(
        "feature_df must contain 'tx_id' or 'wallet_id' column"
    )
