# 下个对话框接力说明

日期：2026-05-22  
项目目录：`D:\release-foundation`  
分支：`phase2-sprint1-burst`

## 当前状态

Sprint 1 主线已经完成并通过主要验收：

```text
CUDA torch: 2.11.0+cu128
torch.cuda.is_available(): True
pytest: 40 passed
ruff: passed on touched eval/posterior files
```

可信输出都在：

```text
outputs/sprint1_*
```

不要用旧的：

```text
outputs/04_curve_posterior/
outputs/05_sbc_multiseed/
outputs/sprint1_run_20260522_104249.log
```

这些旧输出来自 CPU torch 或中断运行，只能当失败记录。

## 已完成结果

Stage 1 `q_phi`：

```text
outputs/sprint1_04_curve_posterior/posterior.pt
50k simulations
device cuda
```

Multi-seed SBC：

```text
outputs/sprint1_05_sbc_multiseed/summary.txt
overall PASS
9/9 params pass c2st_ranks max <= 0.60
```

Stage 2 `r_psi`：

```text
outputs/sprint1_09_descriptor_posterior/posterior.pt
133 high-quality curves
8512 teacher pairs
device cuda
```

Internal leave-one-drug-out：

```text
outputs/sprint1_10_eval_deployment/summary.txt
median R² = +0.6382
mean R²   = +0.4042
median Q0 residual = +0.0000
PASS
```

Internal by-curve 5-fold：

```text
outputs/sprint1_10_eval_deployment/by_curve/summary.txt
median R² = +0.9413
mean R²   = +0.8095
median Q0 residual = +0.0000
```

Cross-DOI 321：

```text
outputs/sprint1_10_eval_deployment/cross_doi_summary.txt
259 high-quality curves of 321
median R² = +0.3502
mean R²   = -0.0033
median Q0 residual = +0.0000
median PASS, mean is warning
```

## 代码修复点

核心修复在 `posterior.py`：

```text
train/load 后显式 estimator.to(device)
sample/log_prob 自动把输入移动到 estimator 实际 device
```

这修复了 `scripts/10_eval_deployment.py` 中 CUDA/CPU mismatch。

`scripts/10_eval_deployment.py --cross-doi` 也已改成尊重 `--device`，不再硬编码 CPU。

## 当前科学结论

可以说：

```text
ADR-026 连续 burst 修复了 Q0 结构性残差；
9 参数 SBI 在 CUDA 上完成重训；
SBC 按项目 primary c2st 标准通过；
内部 LODO 和 cross-DOI median R² 均过线。
```

不能说：

```text
外部分布泛化已经完全解决。
```

原因：

```text
321 cross-DOI median R² = +0.3502 过线，
但 mean R² = -0.0033，说明有失败尾部。
```

## 下一步最稳路线

不要马上换复杂模型。下一步做 321 failure-tail error analysis：

```text
1. 读取 outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv
2. 按 R² 从低到高排序
3. 回连 321 原始 xlsx descriptor
4. 检查失败是否集中在：
   - LA/GA
   - Polymer MW
   - DLC
   - 时间支持长度
   - release shape
   - imputed descriptors: CL Ratio, Drug_Tm, Drug_Pka, Drug_NHA, SA-V, SE
5. 判断 mean R² 负数是少数极端外点，还是系统性外部分布边界
```

建议下个对话框第一句话：

```text
继续 D:\release-foundation。先读 NEXT_CHAT_HANDOFF_ZH.md 和 SPRINT1_RUN_REPORT_ZH.md，然后做 321 cross-DOI failure-tail error analysis，不要先改模型。
```

## 后续探索后现在的主结论（2026-05-23 更新）

321 failure-tail 分析和后续探索已经做过，结论不要再从头猜：

### 1. 当前最强 SBI 候选

不是原始 full SBI，而是：

```text
missingness-aware + weighted distillation
内部简称：weighted_mask1
```

可信外测结果：

```text
outputs/21_eval_plga_fast2x_mask1/cross_doi_summary.txt
median R² = +0.5404
mean R²   = +0.1160
median MAE = 0.1874
```

含义：

```text
这条线对 321 缺失 descriptor 和 fast/short tail 都更稳。
不要再把原始 outputs/sprint1_10_eval_deployment/cross_doi_summary.txt
当唯一 Stage-2 候选。
```

### 2. 现在不要先跳 KAN / RL / 改 ODE

当前证据说明：

```text
主问题不在 ODE scaffold，
也不在“网络不够花”，
而在 descriptor bridge、prefix-aware inference、以及 external tail。
```

### 3. 世界模型探索已经做了三条线

#### A. script23：现有 q_phi 的 partial-observation update（探索性）

脚本：

```text
scripts/23_prefix_world_model_benchmark.py
```

保留结果：

```text
outputs/23_prefix_world_model_benchmark_smoke32_relaxed/summary.txt
```

结论：

```text
当前 q_phi 不是 prefix-aware。
拿 full-curve 训练的 q_phi 直接做 prefix update 会 OOD，
甚至会触发很低的 rejection acceptance。
```

#### B. script24：curve-only 自监督 prefix baseline

脚本：

```text
scripts/24_prefix_curve_ssl_baseline.py
```

可信结果：

```text
outputs/24_prefix_curve_ssl_baseline_full259/summary.txt
```

headline：

```text
1d prefix: median R² = +0.081
3d prefix: median R² = +0.349
7d prefix: median R² = +0.045
```

解释：

```text
“先看部分真实曲线，再预测后段”这条路有信号；
3d prefix 是当前最像样的 sweet spot。
```

#### C. script25：prefix-aware synthetic theta regressor

脚本：

```text
scripts/25_prefix_theta_regressor.py
```

可信结果：

```text
outputs/25_prefix_theta_regressor_full259/summary.txt
```

headline：

```text
1d prefix: median R² = -1.632
3d prefix: median R² = +0.273
7d prefix: median R² = -0.586
```

解释：

```text
prefix-aware latent inference 不是死路，
但当前 7-parameter theta bottleneck 还弱于 curve-only baseline。
```

### 4. 现在最该做什么

不要再回到“只调 zero-shot full SBI”。

主线应改成：

```text
masked-time / partial-observation world model
让模型从全部观测时间点中自己学哪些点重要，
而不是只固定 1d/3d/7d 前缀向量。
```

最合理的下一脚本目标：

```text
script26:
181 完整曲线随机 mask -> 学 partial observations -> full curve reconstruction
测试时在 321 上给真实稀疏观测，预测剩余曲线
```

### 5. 这轮清理后只保留的 prefix 结果目录

已经删掉没有 summary 的 scratch 目录。当前只保留：

```text
outputs/23_prefix_world_model_benchmark_smoke32_relaxed
outputs/24_prefix_curve_ssl_baseline_full259
outputs/25_prefix_theta_regressor_full259
```

## script26 已尝试（2026-05-23 晚）

已新增：

```text
scripts/26_iterative_world_model.py
```

设计：

```text
Transformer observation encoder
+ theta branch within simulator bounds
+ residual curve branch
+ one residual-guided refinement step
+ teacher theta from full-curve q_phi on 181
```

可信结果：

```text
outputs/26_iterative_world_model_v2/summary.txt
```

headline：

```text
1d stage0 median R² = -1.631
3d stage0 median R² = -0.059
7d stage0 median R² = -0.247

1d stage1 median R² = -2.293
3d stage1 median R² = -0.287
7d stage1 median R² = -1.053
```

结论：

```text
这第一版真正的“全部观测 + 一次残差修正”世界模型没有打赢
curve-only SSL baseline（script24）。
目前看来问题不是“世界模型概念错了”，而是：
1) 当前 theta + residual 组合还不够好
2) refinement step 在 external 上会过修正
3) curve-only state 目前比显式 mechanistic state 更稳
```

下一次如果继续做 world model，不要直接再堆更深网络。
优先考虑：

```text
1. 弱化/去掉 theta branch，先做 learned state z + iterative refinement
2. 或把 stage1 update 约束成更小的 correction，避免过修正
3. 不要再把 script26 v2 当成主候选性能线
```
