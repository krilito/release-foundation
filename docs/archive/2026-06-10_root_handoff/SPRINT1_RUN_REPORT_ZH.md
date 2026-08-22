# Sprint 1 运行报告（中文）

日期：2026-05-22  
分支：`phase2-sprint1-burst`

## 一句话结论

这轮 Sprint 1 主线已经跑通，并且主要验收项通过：CUDA 已恢复，连续 burst 的 9 参数模型完成训练、SBC、内部验证、cross-DOI 321 外部验证。  
但不能过度宣称“外部分布泛化完全解决”，因为 321 的 median R² 过线，但 mean R² 略为负数，说明失败尾部还需要单独分析。

## 本轮修复

1. 修复 CUDA 环境问题：

```text
torch 2.11.0+cu128
torch.cuda.is_available() == True
```

2. 清理文档和脚本命令：

```text
不要用 uv run python 跑训练 / SBC / eval
统一用 .\.venv\Scripts\python.exe
```

3. 修复 `DescriptorPosterior` / `CurvePosterior` 的 CUDA device mismatch：

```text
训练或加载后，把 SBI estimator 显式移动到目标 device
sample() / log_prob() 时，把输入移动到 estimator 实际 device
```

这解决了 `scripts/10_eval_deployment.py` 中 CUDA 张量撞 CPU buffer 的崩溃。

4. 更新 `scripts/08_pipeline_sanity.py`：

```text
旧逻辑：q_burst = Q(0)
新逻辑：Q(0)=0，q_burst 通过早期释放曲线体现
```

## 验证结果

代码检查：

```text
pytest: 40 passed
ruff: posterior.py / tests/test_posterior.py / scripts/10_eval_deployment.py passed
```

pipeline sanity：

```text
q_burst KS p = 0.0733 PASS
Q_max   KS p = 0.2611 PASS
verdict: PIPELINE HEALTHY
```

## 训练和评估结果

Stage 1：曲线 posterior `q_phi`

```text
output: outputs/sprint1_04_curve_posterior/posterior.pt
device: cuda
n_simulations: 50000
elapsed: 824.5 s
final_train_loss: -3.1147
final_val_loss:   -2.9471
```

Multi-seed SBC：

```text
output: outputs/sprint1_05_sbc_multiseed/summary.txt
seeds: 10
primary criterion: c2st_ranks max <= 0.60
overall verdict: PASS
```

关键参数全部通过：

```text
q_burst       c2st max = 0.5467 PASS
log_tau_burst c2st max = 0.5767 PASS
Q_max         c2st max = 0.5817 PASS
```

Stage 2：descriptor posterior `r_psi`

```text
output: outputs/sprint1_09_descriptor_posterior/posterior.pt
high-quality curves: 133 / 181
n_pairs: 8512
device: cuda
training: 140.7 s
final_train_loss: -2.8026
final_val_loss:   -2.8521
```

内部 leave-one-drug-out：

```text
output: outputs/sprint1_10_eval_deployment/summary.txt
folds: 21 drugs
curves: 133
median R²: +0.6382
mean R²:   +0.4042
median MAE: 0.1488
median Q0 residual: +0.0000
verdict: PASS
```

5-fold by-curve：

```text
output: outputs/sprint1_10_eval_deployment/by_curve/summary.txt
folds: 5
curves: 133
median R²: +0.9413
mean R²:   +0.8095
median MAE: 0.0622
median Q0 residual: +0.0000
verdict: no regression signal
```

Cross-DOI 321：

```text
output: outputs/sprint1_10_eval_deployment/cross_doi_summary.txt
curves kept: 259 / 321
median R²: +0.3502
mean R²:   -0.0033
median MAE: 0.2067
median Q0 residual: +0.0000
n curves R² > 0:    166 / 259
n curves R² > 0.30: 134 / 259
verdict: median PASS, mean is a warning
```

## 当前目标完成到哪里

已经完成：

- CUDA 修复。
- 9 参数连续 burst ODE 跑通。
- 单元测试和 lint 清理。
- Stage 1 重新训练。
- 10 seed SBC 重新完成。
- Stage 2 descriptor posterior 重新训练。
- 内部 LODO 评估完成。
- by-curve 5-fold 评估完成。
- cross-DOI 321 评估完成。
- 结果文档已写入项目。

尚未完成 / 不能过度宣称：

- cross-DOI 的失败尾部还没有解释。
- 321 数据中缺失的 6 个 descriptor 目前用 181 均值填充，这可能是外部分布失败来源之一。
- SBI rejection sampling 接受率在部分曲线很低，当前能跑完，但后续需要考虑采样策略或后验支持问题。

## 下一步建议

不要马上换复杂模型。下一步应做 321 failure-tail error analysis：

```text
按 R² 排序最差的 321 曲线
检查是否集中在某些 LA/GA、MW、DLC、释放时长、释放形状或 imputed descriptor 区间
判断 mean R² 为负是少数极端外点，还是系统性外部分布问题
```

当前最稳妥的论文/项目表述是：

```text
ADR-026 连续 burst 修复了 Q0 结构性残差，并保持了内部与 median 外部验证性能；
但 cross-DOI 外部分布仍存在尾部失败，需要在候选推荐前做失效边界分析。
```

