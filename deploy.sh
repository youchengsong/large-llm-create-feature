#!/usr/bin/env bash
set -euo pipefail

# KnowFeat 部署脚本 — 将代码同步到远程 GPU 服务器并安装依赖
# 用法:
#   ./deploy.sh                          # 使用默认配置部署
#   ./deploy.sh --host HOST --port PORT  # 指定服务器
#   ./deploy.sh --install-only           # 仅安装依赖（不同步代码）
#   ./deploy.sh --sync-only              # 仅同步代码（不安装依赖）
#   ./deploy.sh --dry-run                # 预览 rsync 操作

# ── 默认配置 ─────────────────────────────────────────
REMOTE_HOST="${DEPLOY_HOST:-connect.westc.seetacloud.com}"
REMOTE_PORT="${DEPLOY_PORT:-25429}"
REMOTE_USER="${DEPLOY_USER:-root}"
REMOTE_DIR="${DEPLOY_DIR:-/root/大模型提取特征}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"

# ── 参数解析 ─────────────────────────────────────────
INSTALL_ONLY=false
SYNC_ONLY=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)       REMOTE_HOST="$2"; shift 2 ;;
        --port)       REMOTE_PORT="$2"; shift 2 ;;
        --user)       REMOTE_USER="$2"; shift 2 ;;
        --dir)        REMOTE_DIR="$2"; shift 2 ;;
        --install-only) INSTALL_ONLY=true; shift ;;
        --sync-only)    SYNC_ONLY=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--host HOST] [--port PORT] [--user USER] [--dir DIR]"
            echo "       [--install-only] [--sync-only] [--dry-run]"
            echo ""
            echo "Environment variables: DEPLOY_HOST, DEPLOY_PORT, DEPLOY_USER, DEPLOY_DIR"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
SSH_CMD="ssh ${SSH_OPTS} -p ${REMOTE_PORT}"
RSYNC_SSH="ssh ${SSH_OPTS} -p ${REMOTE_PORT}"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "═══════════════════════════════════════════════════"
echo "  KnowFeat 部署"
echo "  服务器: ${REMOTE}:${REMOTE_PORT}"
echo "  远程目录: ${REMOTE_DIR}"
echo "  本地目录: ${PROJECT_ROOT}"
echo "═══════════════════════════════════════════════════"

# ── 步骤 0: 连通性检查 ──────────────────────────────
echo ""
echo "[0/3] 检查服务器连通性..."
if ! ${SSH_CMD} "${REMOTE}" "echo ok" >/dev/null 2>&1; then
    echo "✗ 无法连接到 ${REMOTE}:${REMOTE_PORT}"
    echo "  请确认:"
    echo "  1. 服务器实例已启动（seetacloud 实例可能已过期）"
    echo "  2. 连接参数正确（可通过 --host/--port 覆盖或设置 DEPLOY_HOST/DEPLOY_PORT）"
    exit 1
fi
echo "✓ 服务器连接正常"

# ── 步骤 1: 同步代码 ────────────────────────────────
if [[ "${INSTALL_ONLY}" == "false" ]]; then
    echo ""
    echo "[1/3] 同步代码到服务器..."

    ${SSH_CMD} "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"

    RSYNC_FLAGS="-avz --delete --exclude-from=-"
    if [[ "${DRY_RUN}" == "true" ]]; then
        RSYNC_FLAGS="-avzn --delete --exclude-from=-"
        echo "  (dry-run 模式，不实际传输)"
    fi

    rsync ${RSYNC_FLAGS} \
        -e "${RSYNC_SSH}" \
        "${PROJECT_ROOT}/" \
        "${REMOTE}:${REMOTE_DIR}/" <<'EXCLUDES'
.git
.git/
__pycache__
*.pyc
.pytest_cache
.claude-manager
logs/
outputs/single_agent_run/
outputs/baseline_cache.parquet
outputs/baseline_feature_importance.csv
*.egg-info
.tox
.mypy_cache
.coverage
htmlcov
*.swp
*.swo
*~
.DS_Store
deploy.sh
EXCLUDES

    echo "✓ 代码同步完成"
else
    echo ""
    echo "[1/3] 跳过代码同步 (--install-only)"
fi

# ── 步骤 2: 安装依赖 ────────────────────────────────
if [[ "${SYNC_ONLY}" == "false" && "${DRY_RUN}" == "false" ]]; then
    echo ""
    echo "[2/3] 安装 Python 依赖..."

    ${SSH_CMD} "${REMOTE}" bash -s <<INSTALL_SCRIPT
set -euo pipefail
cd '${REMOTE_DIR}'

echo "  Python: \$(python3 --version 2>&1)"
echo "  pip: \$(pip3 --version 2>&1)"

echo "  检查 CUDA..."
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "  CUDA: \$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
fi

echo "  安装依赖 (requirements.txt)..."
pip3 install -r requirements.txt -q 2>&1 | tail -5

echo "  验证关键包..."
python3 -c "
import numpy, pandas, scipy, sklearn, xgboost
import shap, openai
print(f'  numpy={numpy.__version__}')
print(f'  pandas={pandas.__version__}')
print(f'  sklearn={sklearn.__version__}')
print(f'  xgboost={xgboost.__version__}')
print(f'  shap={shap.__version__}')
print(f'  openai={openai.__version__}')
"

echo "  验证 knowfeat 包可导入..."
python3 -c "
import knowfeat
from knowfeat.config import KnowFeatConfig
from knowfeat.data.loader import load_stage2_data
from knowfeat.main import run_stage2
print('  ✓ knowfeat 包导入正常')
"
INSTALL_SCRIPT

    echo "✓ 依赖安装完成"
else
    echo ""
    echo "[2/3] 跳过依赖安装"
fi

# ── 步骤 3: 验证部署 ────────────────────────────────
if [[ "${DRY_RUN}" == "false" && "${INSTALL_ONLY}" == "false" ]]; then
    echo ""
    echo "[3/3] 验证部署..."

    ${SSH_CMD} "${REMOTE}" bash -s <<VERIFY_SCRIPT
set -euo pipefail
cd '${REMOTE_DIR}'

echo "  项目文件:"
echo "    knowfeat/: \$(find knowfeat -name '*.py' | wc -l) 个 Python 文件"
echo "    tests/:    \$(find tests -name '*.py' | wc -l) 个测试文件"
echo "    ModelSet/: \$(find ModelSet -name '*.py' | wc -l) 个模型文件"

if [[ -d data ]]; then
    echo "  数据目录: data/ 存在"
    ls -lh data/*.csv 2>/dev/null | awk '{print "    " \$NF ": " \$5}' || echo "    (无 CSV 文件)"
else
    echo "  ⚠ data/ 目录不存在，需要上传数据文件"
fi

if [[ -d inputs ]]; then
    echo "  知识目录: inputs/ 存在"
else
    echo "  ⚠ inputs/ 目录不存在"
fi

echo ""
echo "  运行快速冒烟测试..."
python3 -c "
from knowfeat.config import KnowFeatConfig
cfg = KnowFeatConfig()
print(f'  配置加载成功: model={cfg.model_name}')
print(f'  数据路径: {cfg.data_dir}')
print(f'  输出路径: {cfg.output_dir}')
"
VERIFY_SCRIPT

    echo ""
    echo "✓ 部署验证完成"
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  部署成功！"
    echo "  运行实验: ssh -p ${REMOTE_PORT} ${REMOTE}"
    echo "    cd ${REMOTE_DIR}"
    echo "    nohup python3 -m knowfeat.main > logs/stage2.log 2>&1 &"
    echo "═══════════════════════════════════════════════════"
else
    echo ""
    echo "[3/3] 跳过验证"
fi
