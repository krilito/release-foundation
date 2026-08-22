# Refactor spec: `scripts/27_partial_curve_npe.py`

Author: Claude (Opus 4.7), session 6
Date: 2026-05-23
Status: Draft — Codex to verify §9 before implementation

---

## 0. 背景

新建 `scripts/27_partial_curve_npe.py`。**不要**修改 `scripts/26_iterative_world_model.py`。

旧 26 的失败原因：拿 full-curve q_phi 的 sample mean 当 teacher，多峰后验取均值得到 unphysical 伪 theta；partial-prefix 后验本来是宽的，蒸馏窄分布迫使学生过度自信；外加无约束 residual 和未被 loss-decreasing 训练的 refinement step。结果 stage0 在 3d 上 median R² = -0.059，stage1 更差，全部输给 baseline 24 的 0.349。

新版思路：**amortized SBI on synthetic partial observations**。学生学的是真实的 `p(theta | prefix)`，不是 q_phi 蒸馏。Script 25 已经做了同源思路的**点估计**版本（3d median R² = 0.273，还是输 24）；script 27 出**完整后验**，区别在于校准过的不确定性，以及能正确报告"前缀信息不够时我不知道"。

参照已有基线：

| Script | 3d median R² on 321 | 备注 |
|---|---|---|
| 24 (curve SSL) | **0.349** | 当前最强基线，27 必须打平 |
| 25 (theta regressor, point) | 0.273 | 输 24 |
| 26 (world model, stage0) | -0.059 | 输 24 |
| 26 (world model, stage1) | -0.287 | 比 stage0 更差 |
| 27 (NPE, this spec) | TBD | 目标 ≥ 0.349 @ 3d，> 0.349 @ 7d |

---

## 1. 必须复用（不要重写）

| 模块 | 文件 | 用法 |
|---|---|---|
| Simulator | `simulator.py` `PLGABiphasic` | `sim.simulate(theta, t_grid)` 返回 `(B, T)` |
| Prior | `PLGABiphasic.prior()` | `sim.sample_prior(n)` 返回 `(n, 9)` |
| t_grid | `configs/plga_phase1.yaml` 读 | `torch.linspace(0, 90, 64)`（ADR-011/012 锁死） |
| NPE 训练 | `from sbi.inference import NPE` | 直接用，不再 wrap |
| Flow + embedding | `from sbi.neural_nets import posterior_nn, embedding_nets` | MAF + FCEmbedding |
| 真曲线读取 | `posterior.py` `interpolate_to_grid()` | 评估时用 |
| 321 加载 + matched_fids 路径 | 照搬 `scripts/26_iterative_world_model.py:611` 周围逻辑 | 不要重新实现 matching |
| Mask 采样 | `scripts/26_iterative_world_model.py` 的 `_sample_train_mask()` 三档 | prefix/random/hybrid，函数级 copy 到 27 |

**不复用** `posterior.CurvePosterior` wrapper —— 它的 conditioning 是 full-curve `Q (T=64)`，27 是 `(Q_masked, mask)` 拼接成 128 维，contract 不同。直接调 `sbi.NPE` 更干净。

---

## 2. 不出现在 27 里

- Teacher theta cache（`outputs/26_iterative_world_model_v2/teacher_theta_cache.pt`）
- Residual head（先证明 NPE 单干能不能打过 24，打过了再考虑加 residual 做 misspecification 修正）
- Refinement / stage1
- 任何来自真实 181 曲线的 theta 监督信号

---

## 3. 训练数据（核心改动）

完全合成、在线生成、不缓存到磁盘：

```python
# 每个 epoch / 每个 batch 重新采
theta = sim.sample_prior(N)            # (N, 9), 9-D, 物理边界 uniform
full  = sim.simulate(theta, t_grid)    # (N, 64), differentiable
mask  = _sample_train_mask(N, T=64)    # (N, 64), 三档 prefix/random/hybrid
noise = 0.03 * torch.randn(N, 64)      # ADR-014 锁死
q_obs = (full + noise) * mask          # masked-out 位置归零

x = torch.cat([q_obs, mask.float()], dim=-1)   # (N, 128) — 给 sbi 的 observation
# theta 是 (N, 9) — 给 sbi 的 parameter target
```

**N 的建议**：先 50_000 跑通（和 `CurvePosterior.train()` 默认一致），SBC 过了再考虑加。

**关键**：x 是 128 维（Q_masked 64 + mask 64）。FCEmbedding 输入维度改成 128，输出仍 16，和 q_phi 一致。这是和 q_phi 唯一的架构差。

---

## 4. 模型

```python
embedding = embedding_nets.FCEmbedding(
    input_dim=128,    # ← 唯一改动
    output_dim=16,
    num_layers=3,
    num_hiddens=64,
)
density_estimator = posterior_nn(
    model="maf",
    embedding_net=embedding,
    hidden_features=64,
    num_transforms=5,
)
inference = NPE(prior=sim.prior(), density_estimator=density_estimator)
inference.append_simulations(theta, x)
inference.train(
    training_batch_size=256,
    max_num_epochs=200,
    validation_fraction=0.1,
    stop_after_epochs=20,
)
posterior = inference.build_posterior()
```

完。无 residual、无 refinement、无第二个 head。

---

## 5. Loss

`sbi.NPE` 内部就是 `-log q(theta | x)`，不需要手写。**不要**外加任何 curve-space MSE 或 theta L2。

---

## 6. 评估（必须三层，缺一不可）

### 6a. NPE 合成 held-out 校准（先看这个，不过别看 321）

留 10_000 条合成 `(theta, x)`。按 1d / 3d / 7d **固定 prefix** 重新生成 mask（而不是训练时的 random mask）做评估：

- **PIT histogram** 每维 theta（应肉眼接近均匀）
- **68% / 95% 后验覆盖率**（应落在 [0.65, 0.71] / [0.92, 0.98]）

输出 `outputs/27_partial_curve_npe/sbc/` 下，PNG + CSV。

**6a 不过就停下**，不要看 321。NPE 自己都没标定，跨域评估没意义。

### 6b. 321 真曲线 suffix R²（必须三档全报）

对每条 321 曲线、每个 prefix（1.0 / 3.0 / 7.0 天）：

1. interpolate 到 64 点 t_grid，按 prefix 时间生成 mask
2. `posterior.sample((1000,), x=...)` 得到 1000 个 theta sample
3. 全部送 simulator → 1000 条预测曲线 → 取**逐点中位数**作为 point prediction
4. 同时记录 **逐点 90% credible band**（5%–95% 分位）
5. 在 prefix **之后**的真实观测时间点上算 R² / MAE

输出到 `outputs/27_partial_curve_npe/summary.txt`，**同表列出 24/25/26 的对应数字**。布局照搬 `outputs/26_iterative_world_model_v2/summary.txt` 的 by-subgroup 切法（fast / short / neither）。

### 6c. 后验宽度 sanity check

对 321 每条曲线、每个 prefix，记录 9 维 theta 后验的 `std(sample)`。画 `mean_std vs prefix_length`。
**预期**：1d > 3d > 7d 单调下降。不单调 → 模型没学到信息增益结构，有 bug。

---

## 7. 验收标准（按 gate 顺序）

| Gate | 标准 | 不过的话 |
|---|---|---|
| G1 | 6a 三档 PIT 肉眼均匀 + 95% 覆盖率 ∈ [0.92, 0.98] | NPE 实现有 bug，先 debug NPE，不要碰 321 |
| G2 | 6c 单调下降 | 编码器没用到 mask 信息，检查输入拼接 |
| G3 | 6b **3d 档 median R² ≥ 0.349**（持平 baseline 24） | 写报告说明为什么 calibrated SBI 不如 curve-only SSL；不要硬上 residual 救场 |
| G4 | 6b **7d 档 median R² 显著 > 24** | 7d 都不行说明 theta 在 321 数据上不可识别，是更深的物理问题（写进 HANDOFF 留给下个 session） |

G3 / G4 不过**不算失败**，只要 G1 / G2 过了，这版就是一次有价值的实验 —— 它给出了"theta 在多大前缀下可识别"的定量答案，这是项目以前没有的诊断。

**补充判据（robustness win）**：baseline 24 的 `3d` headline 是 `median R² = 0.349`，但 explore 报告同时指出其 `mean R²` 被极端 outlier 严重拖负（约 `-21.8`）。因此如果 27 在 `3d` 档上 **median 基本打平 24**，同时 **mean R² 显著更接近 0**，则应在 `summary.txt` 顶部单列说明这是一种**对极端样本更稳的胜利**，不能只按 median headline 把它写成平局。

---

## 8. CLI 约定

照 `scripts/26_iterative_world_model.py` 风格：

- `argparse`
- `--seed`（默认 0，AGENTS.md constraint 3）
- `--config configs/plga_phase1.yaml`
- `--out outputs/27_partial_curve_npe/`
- `--n-train 50000`
- `--noise-sigma 0.03`（ADR-014 默认值，可被 flag 覆盖但默认锁死）
- `_seed_everything()` 覆盖 torch / numpy / random

无 `utils/` / `common/` —— 项目惯例每个 script 自包含，照办。

---

## 9. Codex 开工前需要回答（必填）

1. `matched_fids_csv` 在 `configs/plga_phase1.yaml` 的哪一行？它的实际路径是什么？贴出来。
2. 321 xlsx 的实际加载路径（`scripts/26_iterative_world_model.py:611` 周围）抄给我看，确认 27 用同一份数据。
3. `sim.prior()` 返回的 `torch.distributions.Independent(Uniform(...))` 能直接传给 `sbi.NPE(prior=...)` 吗？sbi 对 prior 类型有要求（需要 `.sample()` 和 `.log_prob()`），verify 一下。如果不行，sbi 提供 `process_prior()` helper，用它包一层。

---

## 10. 实现顺序建议（给 Codex）

1. 回答 §9 的三个问题，贴证据。
2. 写最小可跑骨架：load config → build simulator → sample prior → run NPE.train() → save `posterior.pt`. 不做任何评估。先确认这步能跑通无 NaN。
3. 加 6a SBC 校准。**G1 不过不要往下做**。
4. 加 6b 321 评估和 6c posterior width。
5. 写 `summary.txt`，按 §6b 要求同表列 24/25/26 对比。
6. 把 G1/G2/G3/G4 的实际结果写在 `summary.txt` 顶部作为 verdict block，并显式补一句：如果 `3d median` 只是打平但 `3d mean` 明显优于 baseline 24，按上面的 robustness win 规则单独标注。

---

## 附：上下文文件

- `CLAUDE.md` — 通用编码原则
- `AGENTS.md` — 项目硬约束（种子、单一真值源、机制模块即 simulator subclass）
- `HANDOFF.md` — session 5 close-out，含 ADR-026、Sprint 1 / Phase 2 framing
- `posterior.py:36-384` — `CurvePosterior` 类，NPE wrapper 模板
- `simulator.py:58-243` — `PLGABiphasic`，9-D theta 物理边界在 lines 135-167
- `configs/plga_phase1.yaml:21-27` — t_grid 64 点 / [0, 90] 天的来源
- `outputs/24_prefix_curve_ssl_baseline_full259/summary.txt` — baseline 24 数字
- `outputs/26_iterative_world_model_v2/summary.txt` — script 26 失败数字、subgroup 切法参考
