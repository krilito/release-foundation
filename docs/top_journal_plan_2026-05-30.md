# 顶刊作战计划 — 2026-05-30

目标期刊：**Nature Methods / Nature Machine Intelligence / Nature Communications**
（acceptance ≤ 15%）。下档保底：J. Control. Release / Int. J. Pharmaceutics。

权威关系：本文件是**执行计划（roadmap）**，不覆盖 `docs/paper_requirements_locked_2026-05-28.md`
的硬要求，也不覆盖 `docs/drug_release_closeout_2026-05-28.md` 的 source-of-truth 地位。
若冲突，durable output 文件 > closeout > 本计划。框架冻结决定见 §0。

---

## 0. 计划前提（先认两件事）

1. **故事生死押在壳聚糖湿实验揭盲。** 揭盲前所有工作都是"为那一刀磨刀背"。
2. **框架冻结在 closeout 的朴素版本**（"early-release fingerprint + feasible mechanism-state
   family"）。停写 stack / unified-intelligence 叙事（scripts 80–119 已够）。顶刊新颖性来自
   **方法 + 严格验证 + 前瞻**，不来自"把领域组织成 stack"。这是 R2/R5 hype 警报的硬执行。

---

## 1. 论文论点与四支柱

> Mechanism-constrained Bayesian inference of a feasible release-state family from sparse
> early drug-release observations, with prospective cross-mechanism validation.

| 支柱 | 当前证据 | 缺口 |
|---|---|---|
| **P1** 稀疏早期释放是 OOD 主信号 | ✅ 已测（`outputs/45_input_source_ablation`, R²≈0.94 vs 0.67） | 仅需同口径 CI |
| **P2** 机制路由 theta 中间层 > 匹配 direct-Q | ✅ 已测（`outputs/46`, `outputs/59`，18 cell 全胜） | bootstrap CI（重跑后）+ 对照 NPE |
| **P3** 后验族是**校准**的（不止覆盖率） | ⚠️ 部分（CASP 有覆盖率，OOD 弱；SBC 仅在 curve posterior 05 上跑过） | **SBC 跑到 canonical 方法上** + sharpness |
| **P4** 前瞻预注册跨机制验证 | ⚠️ 已锁未揭盲（`outputs/67`, tag `prospective-chitosan-batch1-2026-05-28`） | **湿实验结果（决定生死）** |

四柱全立 = NM/NMI 候选；缺一掉档。

---

## 2. 证据账本

**已站住（写正文）**：P1、P2、theta=feasible family、RSSM collapse（负）、regime 存在但不解释 benefit（负）。
**已撤回（写进 Methods "试过但不成立"，是 rigor 正资产）**：E1/E4/E9/E11。
**已有半成品（适配即可，非从零）**：
- NPE/SNRE：`scripts/27_partial_curve_npe.py` / `27c_sbi_npe_partial.py` / `27d_residual_npe.py`，
  `sbi>=0.23` 已在依赖；产出 `outputs/27_partial_curve_npe_v1`。**但不在 canonical split、不同口径。**
- SBC：`scripts/05_sbc_curve_posterior.py` / `05b_sbc_multi_seed.py` / `27b_widened_prior_sbc.py`；
  产出 `outputs/05_sbc*`。**但跑在 curve posterior，未跑在论文 canonical 后验上。**
- baseline：`scripts/38_prediction_baselines.py`（含线性插值、MLP、train-median theta）。
**真·从零**：B5 HMC（全仓零引用，需新依赖 numpyro/pyro 或手写 NUTS）；V1–V6 敏感性 harness。
**可复现性硬伤**：A3 未闭环 —— `72_v2` 的 split 在 main 第 504–505 行**硬编码** `random_state=42`，
且其 `load_data()` 取的曲线集（≈181，test 46）与 `data/canonical_split_v1.csv`（150，test 38）**不一致**。

---

## 3. 分阶段执行（每步带验收门）

### Phase 1 — 地基锁定（无新科学风险）
- **T1 闭环 A3**：改 `72_v2` 第 503–505 行，改读 `data/canonical_split_v1.csv`（按 `curve_id`）。
  验收：`72_v2 / 73 / 74b / 76` 打印的 train/cal/test 计数完全一致。
- **T2 重跑锁数字**：RNG 已修的 `73`，连同 `72_v2`、`74b` 重跑，回填文档。
  验收：连续两次运行 bootstrap CI 与 perm p 值**逐位一致**。
- **T3 A2/A4**：每 output 目录有 `lock_metadata.json`；headline 数字清单每行有单一 git-tracked 锚点。

### Phase 2 — 强制 baseline（desk-reject 红线，不能跳）
- **T4 B4 NPE/SNRE 同口径**：把 `27c`/`27d` 适配到 canonical split + 与 theta-route/Active 同评估。
  验收：pooled R² + bootstrap CI 同口径对比表，明确 NPE 赢/平/输。
  **决策门 K4**：NPE 持平或更好 → Active 非新颖点，论文重构为"机制约束 + 前瞻 + 框架"。
- **T5 B5 HMC**：~50 曲线子集做 ground-truth 后验（新脚本 + 新依赖）。
  验收：HMC vs amortized 后验区间重叠 / 覆盖率对照表。
- **T6 B6/B7**：fPCA functional 回归 + mean baseline，进主表底两行。

### Phase 3 — 校准严格性（P3 支柱）
- **T7 S6 SBC on canonical**：把 `05`/`27b` 的 SBC 跑到论文 canonical 后验上。
  验收：rank 直方图 + ECDF，report 是否均匀。
- **T8 S4 sharpness**：cov90/cov50 + mean PI width + naive band 对照。
  验收：覆盖率表必须带宽度列与 naive 对照列。

### Phase 4 — 跨机制 + 前瞻（生死分叉）
- **T9 liposome 诚实化**：n=93，写成 marginal bridge，OOD 覆盖失败进 Limitations（N2）。
- **T10 壳聚糖揭盲**：湿实验 CSV 到手跑 `75`，报预注册 `cov90≥0.83`，一字不改。
  **决策门 K2**：≥0.83 → NM/NMI 路径开；<0.83 → PLGA-only + 诚实迁移极限。

### Phase 5 — 敏感性（防 "by design" 指控）
- **T11 V1–V6 扫描**：粒子数 / 观测预算 / CASP rank r / σ₀ / conformal target / KL weight。
  验收：每 headline 超参单调性曲线，"X under Y, monotone across [low,high]"。

### Phase 6 — 图 + 写作 + 内审
- **T12** 图脚本进 `scripts/figures/`（A6），只消费 outputs；色盲安全、矢量。
- **T13 K5 内审门**：2 独立读者（labmate + 导师）不经提示能否跟上故事；跟不上 → 拆两篇。
- **T14** Methods → Results → Discussion → Intro 成稿。

---

## 4. 决策分叉总图

```
壳聚糖 cov90 ≥0.83 ?
├── 是 ──► NPE 持平/更好 ?
│         ├── 否 ──► [最强] NM/NMI: 机制约束SBI + theta优势 + 校准 + 前瞻跨机制
│         └── 是 ──► [次强] NM/NMI: 框架+前瞻为主, Active 降为 UQ 分支
└── 否 ──► PLGA-only + 诚实迁移极限 ──► Nat Commun (rigor+预注册) 或 J.Control.Release
```

## 5. 关键路径

```
T1(A3闭环) → T2(重跑锁数字) → T4/T5(baseline) → T7(SBC) → (等揭盲T10) → T12-14(图+写作)
壳聚糖湿实验 ──(并行, 外部依赖, deadline最硬)──► 决定最终档位
```
唯一真瓶颈是壳聚糖湿实验的外部时间线 → T4/T5/T7 必须在揭盲前并行做完。

## 6. anti-scope（明确不做）
- ❌ 不扩 80–119 的 snapshot/export/figure/deck。
- ❌ 不写 positioning/stack/battlecard 文档。
- ❌ 不碰 world model / foundation model 措辞。
- ❌ 不为"统一性"再加机制桥，除非某机制 n≥200 独立 held-out。

## 7. 档位预期（诚实）
- 全门过 + 壳聚糖成功：**NM / NMI 有戏**。
- baseline 过但壳聚糖失败：**Nat Commun / 强方法刊**（靠 rigor + 预注册）。
- 跳过 B4/B5/SBC：**SBI 圈 desk-reject，顶刊免谈**。

---

## 8. 需要跑的清单（RUN CHECKLIST）

类型：`RUN`=现有脚本直接跑；`BUILD+RUN`=先改/新写代码再跑；`BLOCKED`=等外部数据。

| ID | 动作 | 命令 / 产物 | 类型 | 验收门 |
|---|---|---|---|---|
| R0 | 生成 canonical split | `python scripts/make_canonical_split.py` → `data/canonical_split_v1.csv` | RUN（已完成） | n=150/89/23/38 |
| R1 | A3 闭环：72_v2 读固定 split | 改 `72_v2` L503–505 → 跑 `python scripts/72_canonical_benchmark_v2.py --seed 0` | BUILD+RUN | 各脚本 split 计数一致 |
| R2 | 重跑诊断锁数字 | `python scripts/73_diagnostics.py --seed 42` | RUN | 两次运行 p/CI 逐位一致 |
| R3 | 重跑 regime 验证 | `python scripts/74b_regime_verification_v2.py` | RUN | 同上（注：74b 未接 --seed，需补 A1） |
| R4 | 审计现有 NPE 现状 | `python scripts/27c_sbi_npe_partial.py`（看当前 split/口径） | RUN | 明确它现在跑在哪份 split |
| R5 | B4：NPE/SNRE 同口径头对头 | 适配 `27c`/`27d` 到 canonical split + 同评估 → 新 `outputs/` | BUILD+RUN | pooled R²+CI 对比表（K4） |
| R6 | 审计现有 SBC 现状 | `python scripts/05_sbc_curve_posterior.py` | RUN | 明确它跑在哪个后验 |
| R7 | B5 HMC ground-truth | 新脚本 + numpyro/pyro 依赖；~50 曲线子集 | BUILD+RUN | HMC vs amortized 覆盖对照 |
| R8 | B6/B7 fPCA + mean baseline | 扩 `38_prediction_baselines.py` | BUILD+RUN | 进主表底两行 |
| R9 | S6 SBC on canonical 后验 | 适配 `05`/`27b` 到论文 canonical 方法 | BUILD+RUN | rank 均匀性直方图/ECDF |
| R10 | S4 sharpness 列 | 扩 conformal export（`87_*`）加 width + naive band | BUILD+RUN | 覆盖率表含宽度/naive 列 |
| R11 | V1–V6 敏感性扫描 | 新 sweep harness | BUILD+RUN | 单调性曲线 |
| R12 | 壳聚糖揭盲 | `python scripts/75_chitosan_prospective_eval.py --observed-csv <wetlab.csv>` | BLOCKED | 报 cov90≥0.83（K2） |

**优先级建议**：R1 → R2/R3 → R4 → R5 → R6 → R9 → R7 → R8/R10/R11，全部在 R12 揭盲前并行推完。

---

**Owner**: 项目负责人
**Status**: Draft，待负责人确认后转 active
**关联**: ADR-031（canonical 脚本）、`paper_requirements_locked_2026-05-28.md`、`drug_release_closeout_2026-05-28.md`
