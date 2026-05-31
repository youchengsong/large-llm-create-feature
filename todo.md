# ICDM 2026 论文实验 TODO

> KnowFeat: Knowledge-Traceable Feature Engineering
>
> **最后更新：2026-05-31**

---

## Loop 模式执行规则

1. 从上到下找到第一个未完成的 `- [ ]` 任务，执行它
2. 可并行的任务（标注了"可并行"）同时启动，用 nohup 后台运行，定期检查完成状态
3. 完成后勾选 `- [x]`，commit + push，然后继续下一个任务
4. **所有远程命令必须用 nohup/后台运行，不要在前台阻塞**
5. 如果某个任务失败，记录错误信息到日志，跳过继续下一个，最后汇报失败项

---

## 服务器信息

### SSH 连接

```bash
# 机器A (simecny 数字人民币实验)
sshpass -p '1LT29DbqKjKY' ssh -p 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com

# 机器B (公开数据集: credit_g, diabetes, blood_transfusion)
sshpass -p 'M7juUy9FhtmM' ssh -p 34431 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com

# 机器C (公开数据集: heart, bank_marketing, breast_w)
sshpass -p 'IAo8KtG1J4H9' ssh -p 33982 -o StrictHostKeyChecking=no root@connect.bjb1.seetacloud.com
```

### SSH 命令模板

```bash
# 执行远程命令（短命令）
sshpass -p '<密码>' ssh -p <端口> -o StrictHostKeyChecking=no root@<主机> "export PATH=/root/miniconda3/bin:\$PATH && export KNOWFEAT_API_KEY=sk-2e921b3293694512bfc088f0b9251900 && export KNOWFEAT_BASE_URL=https://api.deepseek.com/v1 && cd /root/knowfeat && <命令>"

# 执行远程后台任务（长时间运行）
sshpass -p '<密码>' ssh -p <端口> -o StrictHostKeyChecking=no root@<主机> "export PATH=/root/miniconda3/bin:\$PATH && export KNOWFEAT_API_KEY=sk-2e921b3293694512bfc088f0b9251900 && export KNOWFEAT_BASE_URL=https://api.deepseek.com/v1 && cd /root/knowfeat && nohup <命令> > logs/<日志名>.log 2>&1 &"

# 检查后台任务是否完成
sshpass -p '<密码>' ssh -p <端口> -o StrictHostKeyChecking=no root@<主机> "ps aux | grep python | grep -v grep"

# 查看日志
sshpass -p '<密码>' ssh -p <端口> -o StrictHostKeyChecking=no root@<主机> "tail -20 /root/knowfeat/logs/<日志名>.log"

# 拉取结果文件到本机
sshpass -p '<密码>' scp -P <端口> -o StrictHostKeyChecking=no root@<主机>:/root/knowfeat/outputs/eval_results/*.json /home/ubuntu/Projects/大模型提取特征/outputs/eval_results/

# 推送文件到服务器
cat <本地文件> | sshpass -p '<密码>' ssh -p <端口> -o StrictHostKeyChecking=no root@<主机> "cat > /root/knowfeat/<远程路径>"
```

### 环境信息

- **项目目录**: `/root/knowfeat/`
- **Python**: `/root/miniconda3/bin/python`（PATH已配置在~/.bashrc）
- **DeepSeek API Key**: `sk-2e921b3293694512bfc088f0b9251900`
- **DeepSeek Base URL**: `https://api.deepseek.com/v1`
- **DeepSeek Model**: `deepseek-chat`
- 3 台机器共用 1 个 DeepSeek 账号（实际并发低，不会限速）

---

## Phase 0.9: 服务器部署 ✅

- [x] **P0.9a** 生成 `requirements.txt` ✅
- [x] **P0.9b** 部署 3 台服务器环境 ✅
  - 代码和数据已通过 tarball 传输到 3 台服务器
  - 依赖已安装（xgboost, scikit-learn, openai, openfe, autofeat, etc.）
  - 环境变量已配置
  - 验证通过: breast_w No-FE AUC=0.9926

---

## Phase 1: Exp 1 — 主性能对比 (Performance Comparison)

> 目标：7个数据集 × 6种方法 → 两张性能对比表
> 结果存放: 各机器 `/root/knowfeat/outputs/eval_results/<dataset>_<method>.json`

### 步骤 1.1: 三台机器并行启动 No-FE + CPU方法（可并行）

> 以下任务在3台机器上**同时**启动

- [ ] **T1.1** 机器A: simecny No-FE + OpenFE + AutoFeat（3个CPU任务可并行）
  ```bash
  # A: No-FE
  sshpass -p '1LT29DbqKjKY' ssh -p 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && nohup python scripts/evaluate.py --dataset simecny --method No-FE --baseline > logs/t1.1_nfe.log 2>&1 &"
  # A: OpenFE
  sshpass -p '1LT29DbqKjKY' ssh -p 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && nohup python scripts/run_openfe.py --dataset simecny --n-features 30 > logs/t1.1_openfe.log 2>&1 &"
  # A: AutoFeat
  sshpass -p '1LT29DbqKjKY' ssh -p 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && nohup python scripts/run_autofeat.py --dataset simecny > logs/t1.1_autofeat.log 2>&1 &"
  ```

- [ ] **T1.2** 机器B: credit_g + diabetes + blood_transfusion 的 No-FE + OpenFE + AutoFeat
  ```bash
  # B: 3个数据集的 No-FE
  sshpass -p 'M7juUy9FhtmM' ssh -p 34431 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && python scripts/evaluate.py --dataset credit_g --method No-FE --baseline > logs/t1.2_nfe_credit.log 2>&1 && python scripts/evaluate.py --dataset diabetes --method No-FE --baseline > logs/t1.2_nfe_diabetes.log 2>&1 && python scripts/evaluate.py --dataset blood_transfusion --method No-FE --baseline > logs/t1.2_nfe_blood.log 2>&1 &"
  # B: 3个数据集的 OpenFE（串行，避免内存争抢）
  sshpass -p 'M7juUy9FhtmM' ssh -p 34431 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && nohup bash -c 'python scripts/run_openfe.py --dataset credit_g && python scripts/run_openfe.py --dataset diabetes && python scripts/run_openfe.py --dataset blood_transfusion' > logs/t1.2_openfe.log 2>&1 &"
  # B: 3个数据集的 AutoFeat
  sshpass -p 'M7juUy9FhtmM' ssh -p 34431 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && nohup bash -c 'python scripts/run_autofeat.py --dataset credit_g && python scripts/run_autofeat.py --dataset diabetes && python scripts/run_autofeat.py --dataset blood_transfusion' > logs/t1.2_autofeat.log 2>&1 &"
  ```

- [ ] **T1.3** 机器C: heart + bank_marketing + breast_w 的 No-FE + OpenFE + AutoFeat
  ```bash
  # C: 3个数据集的 No-FE
  sshpass -p 'IAo8KtG1J4H9' ssh -p 33982 -o StrictHostKeyChecking=no root@connect.bjb1.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && python scripts/evaluate.py --dataset heart --method No-FE --baseline > logs/t1.3_nfe_heart.log 2>&1 && python scripts/evaluate.py --dataset bank_marketing --method No-FE --baseline > logs/t1.3_nfe_bank.log 2>&1 && python scripts/evaluate.py --dataset breast_w --method No-FE --baseline > logs/t1.3_nfe_breast.log 2>&1 &"
  # C: 3个数据集的 OpenFE
  sshpass -p 'IAo8KtG1J4H9' ssh -p 33982 -o StrictHostKeyChecking=no root@connect.bjb1.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && nohup bash -c 'python scripts/run_openfe.py --dataset heart && python scripts/run_openfe.py --dataset bank_marketing && python scripts/run_openfe.py --dataset breast_w' > logs/t1.3_openfe.log 2>&1 &"
  # C: 3个数据集的 AutoFeat
  sshpass -p 'IAo8KtG1J4H9' ssh -p 33982 -o StrictHostKeyChecking=no root@connect.bjb1.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && nohup bash -c 'python scripts/run_autofeat.py --dataset heart && python scripts/run_autofeat.py --dataset bank_marketing && python scripts/run_autofeat.py --dataset breast_w' > logs/t1.3_autofeat.log 2>&1 &"
  ```

### 步骤 1.2: CPU方法完成后，评估CPU方法 + 启动LLM方法

> 等 T1.1~T1.3 的 OpenFE/AutoFeat 完成后执行
> LLM方法（CAAFE、FeatLLM、KnowFeat）必须**串行**执行（避免API限速）

- [ ] **T1.4** 机器A: 评估 OpenFE/AutoFeat 结果 + 串行跑 CAAFE → FeatLLM → KnowFeat
  ```bash
  # A: 评估 OpenFE
  sshpass -p '1LT29DbqKjKY' ssh -p 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && python scripts/evaluate.py --dataset simecny --method OpenFE --features-csv outputs/openfe/simecny_features.csv"
  # A: 评估 AutoFeat
  sshpass -p '1LT29DbqKjKY' ssh -p 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && python scripts/evaluate.py --dataset simecny --method AutoFeat --features-csv outputs/autofeat/simecny_features.csv"
  # A: 串行跑3个LLM方法
  sshpass -p '1LT29DbqKjKY' ssh -p 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && export KNOWFEAT_API_KEY=sk-2e921b3293694512bfc088f0b9251900 && export KNOWFEAT_BASE_URL=https://api.deepseek.com/v1 && cd /root/knowfeat && nohup bash -c 'python scripts/run_caafe.py --dataset simecny --iterations 10 && python scripts/evaluate.py --dataset simecny --method CAAFE --features-csv outputs/caafe/simecny_features.csv && python scripts/run_featllm.py --dataset simecny && python scripts/evaluate.py --dataset simecny --method FeatLLM --features-csv outputs/featllm/simecny_features.csv && python -m knowfeat.main --stage stage2 --max-features 30 && python scripts/evaluate.py --dataset simecny --method KnowFeat --features-csv outputs/stage2_features.csv' > logs/t1.4_llm.log 2>&1 &"
  ```

- [ ] **T1.5** 机器B: 评估 OpenFE/AutoFeat + 串行跑 LLM 方法（credit_g → diabetes → blood_transfusion）
  ```bash
  # B: 评估已完成的CPU方法
  sshpass -p 'M7juUy9FhtmM' ssh -p 34431 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && for ds in credit_g diabetes blood_transfusion; do python scripts/evaluate.py --dataset \$ds --method OpenFE --features-csv outputs/openfe/\${ds}_features.csv; python scripts/evaluate.py --dataset \$ds --method AutoFeat --features-csv outputs/autofeat/\${ds}_features.csv; done"
  # B: 串行跑LLM方法
  sshpass -p 'M7juUy9FhtmM' ssh -p 34431 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && export KNOWFEAT_API_KEY=sk-2e921b3293694512bfc088f0b9251900 && export KNOWFEAT_BASE_URL=https://api.deepseek.com/v1 && cd /root/knowfeat && nohup bash -c 'for ds in credit_g diabetes blood_transfusion; do python scripts/run_caafe.py --dataset \$ds && python scripts/evaluate.py --dataset \$ds --method CAAFE --features-csv outputs/caafe/\${ds}_features.csv && python scripts/run_featllm.py --dataset \$ds && python scripts/evaluate.py --dataset \$ds --method FeatLLM --features-csv outputs/featllm/\${ds}_features.csv && python scripts/run_knowfeat_public.py --dataset \$ds && python scripts/evaluate.py --dataset \$ds --method KnowFeat --features-csv outputs/\${ds}/stage2_features.csv; done' > logs/t1.5_llm.log 2>&1 &"
  ```

- [ ] **T1.6** 机器C: 评估 OpenFE/AutoFeat + 串行跑 LLM 方法（heart → bank_marketing → breast_w）
  ```bash
  # C: 评估已完成的CPU方法
  sshpass -p 'IAo8KtG1J4H9' ssh -p 33982 -o StrictHostKeyChecking=no root@connect.bjb1.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && cd /root/knowfeat && for ds in heart bank_marketing breast_w; do python scripts/evaluate.py --dataset \$ds --method OpenFE --features-csv outputs/openfe/\${ds}_features.csv; python scripts/evaluate.py --dataset \$ds --method AutoFeat --features-csv outputs/autofeat/\${ds}_features.csv; done"
  # C: 串行跑LLM方法
  sshpass -p 'IAo8KtG1J4H9' ssh -p 33982 -o StrictHostKeyChecking=no root@connect.bjb1.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && export KNOWFEAT_API_KEY=sk-2e921b3293694512bfc088f0b9251900 && export KNOWFEAT_BASE_URL=https://api.deepseek.com/v1 && cd /root/knowfeat && nohup bash -c 'for ds in heart bank_marketing breast_w; do python scripts/run_caafe.py --dataset \$ds && python scripts/evaluate.py --dataset \$ds --method CAAFE --features-csv outputs/caafe/\${ds}_features.csv && python scripts/run_featllm.py --dataset \$ds && python scripts/evaluate.py --dataset \$ds --method FeatLLM --features-csv outputs/featllm/\${ds}_features.csv && python scripts/run_knowfeat_public.py --dataset \$ds && python scripts/evaluate.py --dataset \$ds --method KnowFeat --features-csv outputs/\${ds}/stage2_features.csv; done' > logs/t1.6_llm.log 2>&1 &"
  ```

### 步骤 1.3: 回收结果并汇总

> 等 T1.4~T1.6 全部完成后执行

- [ ] **T1.7** 从 3 台机器拉取 eval_results 到本机
  ```bash
  # 从A拉取
  sshpass -p '1LT29DbqKjKY' scp -P 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com:"/root/knowfeat/outputs/eval_results/*.json" /home/ubuntu/Projects/大模型提取特征/outputs/eval_results/
  # 从B拉取
  sshpass -p 'M7juUy9FhtmM' scp -P 34431 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com:"/root/knowfeat/outputs/eval_results/*.json" /home/ubuntu/Projects/大模型提取特征/outputs/eval_results/
  # 从C拉取
  sshpass -p 'IAo8KtG1J4H9' scp -P 33982 -o StrictHostKeyChecking=no root@connect.bjb1.seetacloud.com:"/root/knowfeat/outputs/eval_results/*.json" /home/ubuntu/Projects/大模型提取特征/outputs/eval_results/
  ```

- [ ] **T1.8** 编写 `scripts/gen_table1_table2.py` 生成论文表格
  - 读取 `outputs/eval_results/*.json`
  - 生成 `outputs/tables/table1_simecny.csv`（数字人民币: 6方法 × Recall/Precision/F1/AUC）
  - 生成 `outputs/tables/table2_public.csv`（公开数据集: 6方法 × 6数据集 × AUC/Accuracy）

---

## Phase 2: Exp 2 — 知识消融 (Knowledge Ablation)

> 仅在数字人民币上运行，量化每种知识资产的边际贡献
> **在机器 A 上串行执行（Phase 1 完成后）**
> 6个配置必须串行（都需要API）

- [ ] **T2.1** 机器A: 串行跑 6 个消融配置
  ```bash
  sshpass -p '1LT29DbqKjKY' ssh -p 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com "export PATH=/root/miniconda3/bin:\$PATH && export KNOWFEAT_API_KEY=sk-2e921b3293694512bfc088f0b9251900 && export KNOWFEAT_BASE_URL=https://api.deepseek.com/v1 && cd /root/knowfeat && nohup bash -c '
  # Full knowledge
  python -m knowfeat.main --stage stage2 --max-features 30 2>&1 | tee logs/t2_full.log
  mkdir -p outputs/ablation/full && cp outputs/stage2_features.csv outputs/ablation/full/
  python scripts/evaluate.py --dataset simecny --method KnowFeat-Full --features-csv outputs/ablation/full/stage2_features.csv
  # Rules only
  python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter rules 2>&1 | tee logs/t2_rules.log
  mkdir -p outputs/ablation/rules && cp outputs/stage2_features.csv outputs/ablation/rules/
  python scripts/evaluate.py --dataset simecny --method KnowFeat-Rules --features-csv outputs/ablation/rules/stage2_features.csv
  # Indicators only
  python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter indicators 2>&1 | tee logs/t2_indicators.log
  mkdir -p outputs/ablation/indicators && cp outputs/stage2_features.csv outputs/ablation/indicators/
  python scripts/evaluate.py --dataset simecny --method KnowFeat-Indicators --features-csv outputs/ablation/indicators/stage2_features.csv
  # Expert only
  python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter expert 2>&1 | tee logs/t2_expert.log
  mkdir -p outputs/ablation/expert && cp outputs/stage2_features.csv outputs/ablation/expert/
  python scripts/evaluate.py --dataset simecny --method KnowFeat-Expert --features-csv outputs/ablation/expert/stage2_features.csv
  # Court only
  python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter court 2>&1 | tee logs/t2_court.log
  mkdir -p outputs/ablation/court && cp outputs/stage2_features.csv outputs/ablation/court/
  python scripts/evaluate.py --dataset simecny --method KnowFeat-Court --features-csv outputs/ablation/court/stage2_features.csv
  # Schema only (no domain knowledge)
  python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter schema_only 2>&1 | tee logs/t2_schema.log
  mkdir -p outputs/ablation/schema_only && cp outputs/stage2_features.csv outputs/ablation/schema_only/
  python scripts/evaluate.py --dataset simecny --method KnowFeat-Schema --features-csv outputs/ablation/schema_only/stage2_features.csv
  ' > logs/t2_all.log 2>&1 &"
  ```

- [ ] **T2.2** 拉取消融结果到本机
  ```bash
  sshpass -p '1LT29DbqKjKY' scp -P 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com:"/root/knowfeat/outputs/eval_results/simecny_KnowFeat-*.json" /home/ubuntu/Projects/大模型提取特征/outputs/eval_results/
  ```

- [ ] **T2.3** 编写 `scripts/gen_table3.py` 生成消融表格
  - `outputs/tables/table3_ablation.csv`（6配置 × Recall/F1/AUC/通过率）

---

## Phase 3: Exp 3 — 溯源完整性评估 (Traceability Metrics)

> 依赖: Phase 1 中 CAAFE、FeatLLM、KnowFeat 在 simecny 上的运行产物
> 在本机执行（结果已拉回）

- [ ] **T3.1** 从机器A拉取溯源产物
  ```bash
  sshpass -p '1LT29DbqKjKY' scp -P 52881 -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com:"/root/knowfeat/outputs/stage2_provenance.json" /home/ubuntu/Projects/大模型提取特征/outputs/
  sshpass -p '1LT29DbqKjKY' scp -P 52881 -o StrictHostKeyChecking=no -r root@connect.westc.seetacloud.com:"/root/knowfeat/outputs/caafe/" /home/ubuntu/Projects/大模型提取特征/outputs/
  sshpass -p '1LT29DbqKjKY' scp -P 52881 -o StrictHostKeyChecking=no -r root@connect.westc.seetacloud.com:"/root/knowfeat/outputs/featllm/" /home/ubuntu/Projects/大模型提取特征/outputs/
  ```

- [ ] **T3.2** 编写 `scripts/compute_traceability.py` 计算4个溯源指标
  - 知识引用率、知识可验证率、推理步骤覆盖率、字段溯源率
  - 输出: `outputs/eval_results/traceability_metrics.json`

- [ ] **T3.3** 生成 Table 4: `outputs/tables/table4_traceability.csv`

---

## Phase 4: Exp 4 — 专家打分 (Expert Evaluation)

> ⚠️ 人工任务，loop 模式跳过

- [ ] **T4.1** 编写 `scripts/prepare_expert_eval.py` 生成匿名化评估材料
- [ ] **T4.2** 联系 3-5 名专家评分（人工）
- [ ] **T4.3** 编写 `scripts/summarize_expert_eval.py` 汇总评分

---

## Phase 5: Exp 5 — Case Study

> 依赖: Phase 1 中三个方法的运行产物

- [ ] **T5.1** 编写 `scripts/gen_case_study.py` 提取 2-3 个代表性特征并排对比
- [ ] **T5.2** 生成 Figure 3

---

## Phase 6: Exp 6 — 效率对比 (Efficiency)

> 在数字人民币上对比各方法的计算成本

- [ ] **T6.1** 编写 `scripts/gen_efficiency_table.py` 汇总运行指标
  - 总运行时间、API 调用次数、API 费用、生成/通过特征数、通过率
  - 输出: `outputs/tables/table6_efficiency.csv`

---

## 执行顺序与依赖

```
Phase 0.9 (服务器部署) ✅ 完成
  │
  ├── T1.1 + T1.2 + T1.3 (3台机器并行: No-FE + CPU方法)
  │     ↓ 等CPU方法完成
  ├── T1.4 + T1.5 + T1.6 (3台机器并行: 评估CPU + LLM方法串行)
  │     ↓ 等所有完成
  ├── T1.7 (拉取结果) → T1.8 (生成Table 1&2)
  │
  ├── T2.1 (机器A: 消融实验串行) → T2.2 (拉取) → T2.3 (Table 3)
  │
  ├── T3.1 (拉取溯源) → T3.2 (计算指标) → T3.3 (Table 4)
  ├── T5.1 → T5.2 (Case Study)
  └── T6.1 (效率表)
        │
  Phase 4 (专家打分) ← 人工，最后
```

## No-FE Baseline 参考值

| 数据集 | AUC | Accuracy |
|--------|-----|----------|
| credit_g | 0.7670 | 0.7350 |
| diabetes | 0.8027 | 0.7558 |
| blood_transfusion | 0.6824 | 0.7547 |
| heart | 0.9320 | 0.8598 |
| bank_marketing | 0.9330 | 0.9082 |
| breast_w | 0.9926 | 0.9664 |
