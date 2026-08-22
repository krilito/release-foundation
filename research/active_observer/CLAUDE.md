# Active Observer Research — Constraints

## What This Is

Particle Active Kinetic Observer for PLGA controlled release.
核心算法：formulation → kinetic prior particles → ODE rollout → 信息量选点 → 一次观测 posterior update → 全曲线预测。

## Hard Constraints

### 1. ODE 只用 PLGABiphasic

`simulate_plga_ode` 必须接到 `simulator.py` 中的 `PLGABiphasic.simulate_numpy()`。
禁止自己写 ODE solver，禁止用 surrogate 替代物理 ODE。

### 2. Theta 布局锁定

```
[log_kw, log_kh, log_alpha, log_kd, log_ke, m_crit, q_burst, log_tau_burst, Q_max]
```

9 维，顺序不可变。和 `PLGABiphasic.param_names` 一致。

### 3. 不改 simulator.py

本研究只读 simulator，不修改。如需新 ODE 形式，开新文件。

### 4. Prior 粒子来源

prior 由三部分组成：
- ExtraTrees 树级预测（tree particles）
- KNN 相似配方检索（knn particles）
- 高斯 jitter（基于 residual std）

禁止用 neural network 做 prior（那是 Phase 2+ 的事）。

### 5. Posterior 只做 reweighting

posterior update = likelihood reweighting，不重新拟合 theta，不跑 MCMC。
这是 particle filter 的标准做法，保持可解释性。

### 6. 评估指标固定

必须报告：
- `future_rmse_mean` / `future_rmse_median`
- `coverage_90` / `coverage_80`
- `width_90` / `width_80`
- `crps`
- `ess_after`

禁止只报 RMSE 不报 uncertainty 指标。

### 7. Baseline 对比固定

必须包含：
- `zero_early_prior_only`（0-early prior）
- `fixed_Xd_posterior`（每个候选时间点各一个）
- `active_one_point_posterior`（active 选点）
- `directQ_one_point_*`（端到端 baseline）

禁止只跑 active 不跑 baseline。

## Known Issues

### Prior 高估 Q_max（已修复）

prior 系统性高估释放量 ~15%。根因：oracle Q_max 偏向 1.0。

修复：用实际 plateau 分布（mean=0.85, std=0.18）替代 oracle Q_max（mean=0.95, std=0.06）。
加上 conformal calibration，coverage 从 0.38 跳到 0.90+。

### 小 polymer family coverage 低

PLA-co-PALA/PLGA-co-PALA 只有 4 条曲线，coverage 只有 0.70-0.75。
这是数据量问题，不是方法问题。主要 family（PLGA/PVL-co-PAVL/PCL）coverage > 0.98。

### Direct-Q 点预测仍领先

Direct-Q RMSE=0.120 vs active two-point RMSE=0.122。
但 Direct-Q 没有 uncertainty 量化，posterior 方法有 calibrated intervals。

## File Map

| 文件 | 用途 |
|------|------|
| `scripts/68_active_kinetic_observer.py` | 主脚本（--groupkfold 模式） |
| `scripts/68_prepare_data.py` | 数据准备 |
| `scripts/68_plot_active_observer.py` | 可视化 |
| `data/formulations.csv` | 150 条曲线配方特征 |
| `data/curves_long.csv` | 释放曲线 long format |
| `data/theta_bank.csv` | oracle 拟合 theta |
| `outputs_active_observer/` | 结果 CSV + figures/ |
| `research/active_observer/` | 本目录：研究文档 |

## Next Steps

1. [x] Two-point sequential observer ✓
2. [x] GroupKFold 评估 ✓
3. [ ] Coverage_80 单独校准
4. [ ] 和 SBI NPE 对比
5. [ ] 更大测试集验证
