"""计算基线+2个新特征的组合PR-AUC"""
import pandas as pd, numpy as np, json, sys, os, tempfile, importlib.util
sys.path.insert(0, '/root/大模型提取特征')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import xgboost as xgb

def calc_prauc(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, te in skf.split(X, y):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, tree_method='hist', random_state=42, eval_metric='aucpr')
        m.fit(X[tr], y[tr], verbose=False)
        scores.append(average_precision_score(y[te], m.predict_proba(X[te])[:,1]))
    return np.mean(scores)

# 1. 直接跑 feature_eng 拿到完整特征矩阵（保留 From_Account / To_Account）
spec = importlib.util.spec_from_file_location("fe", "/root/大模型提取特征/inputs/baseline/feature_engineering.py")
fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)

with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "feat.csv")
    fe.feature_eng("data/transactions.csv", "data/accounts.csv", out, close_path="data/close.csv")
    orig = os.path.join(tmp, "feat_original.csv")
    df = pd.read_csv(orig) if os.path.exists(orig) else pd.read_csv(out)

print("rows:", len(df), "cols:", len(df.columns))

# 2. 确定特征列和标签
exclude = {'is_laundering','id','Timestamp','From_Account','To_Account','tx_id','timestamp','src','dst','is_risk','amount'}
feat_cols = [c for c in df.columns if c not in exclude]
y = df['is_laundering'].values
print("baseline features:", len(feat_cols))

# 3. 基线 PR-AUC
bp = calc_prauc(df[feat_cols].values, y)
print("Baseline PR-AUC: %.6f (%d features)" % (bp, len(feat_cols)))

# 4. 从原始数据恢复 From_Account/To_Account（feature_eng 会删掉它们）
tx_raw = pd.read_csv("data/transactions.csv")
tx_raw['Timestamp'] = pd.to_datetime(tx_raw['timestamp'])
tx_sorted = tx_raw.sort_values('Timestamp').reset_index(drop=True)
df['From_Account'] = tx_sorted['src'].values[:len(df)]
df['To_Account'] = tx_sorted['dst'].values[:len(df)]
print("Restored From_Account/To_Account from raw data")

# 5. 计算 2 个新特征并合并
with open("output/features.json") as f:
    features = json.load(f)

for feat in features:
    name = feat['feature_name']
    ns = {'tx_df': tx_raw.copy(), 'pd': pd, 'np': np}
    exec(feat['code'], ns)
    r = ns['result']
    val_col = [c for c in r.columns if c != 'wallet_id'][0]
    mapping = r.set_index('wallet_id')[val_col]
    df[name + '_src'] = df['From_Account'].map(mapping).fillna(0)
    df[name + '_dst'] = df['To_Account'].map(mapping).fillna(0)
    feat_cols += [name + '_src', name + '_dst']
    print("  added:", name)

# 5. 组合 PR-AUC
cp = calc_prauc(df[feat_cols].values, y)
print("Combined PR-AUC: %.6f (%d features)" % (cp, len(feat_cols)))
print("\n===== RESULT =====")
print("Baseline:  %.6f" % bp)
print("Combined:  %.6f" % cp)
print("Delta:     %+.6f" % (cp - bp))
print("Improve:   %+.2f%%" % ((cp - bp) / bp * 100))
