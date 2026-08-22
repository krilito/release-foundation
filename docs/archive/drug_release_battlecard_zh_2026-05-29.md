# 药物释放竞品 / 借鉴 / 防守 Battlecard（中文）

日期：`2026-05-29`

目的：

```text
把外部对象压成一张真正可执行的中文作战图：
1. 谁要正面对打
2. 谁主要拿来借
3. 谁需要防守叙事
4. 每一类为什么这样处理
5. 我们当前有哪些本地证据能支撑这个判断
```

## 1. 总判断

外部赛道现在不是“一个对手”，而是三种不同压力：

1. `benchmark rival`
2. `platform threat`
3. `infrastructure / workflow borrowing target`

最重要的一句是：

`不要把所有外部工作都当成同一种竞品。`

因为这三类东西抢的是三个不同位置：

- benchmark rival 抢的是“谁预测更好”
- platform threat 抢的是“谁更有用”
- infrastructure/workflow 抢的是“谁定义标准输入和标准流程”

## 2. 一页表版本

| 对象 | 类别 | 现在主要在做什么 | 对我们最直接的威胁/价值 | 我们该怎么对待 |
| --- | --- | --- | --- | --- |
| `NC / Bannigan` | `fight` | few-shot / zero-shot release forecasting | reviewer 默认 benchmark 对手 | 正面对打 |
| explainable LAI few-shot papers | `fight` | early-point + explainable tabular ML | 更容易讲清楚的 release predictor | 正面对打 |
| 新的更大规模 PLGA supervised predictor | `fight` | 更大数据上的 direct release prediction | 容易让我们显得只是“另一个 PLGA 模型” | 正面对打 |
| liposome IVR workflow | `borrow` | non-PLGA assay / workflow discipline | 帮我们把 bridge 做真实 | 重点借 |
| Scientific Data / AI-ready release data | `borrow` | schema / metadata / benchmark packaging | 帮我们把 unified stack 做成 field-standard 味道 | 重点借 |
| computational pharmaceutics / hybrid modeling | `borrow` | 机制 + ML 混合科学语言 | 帮我们避免掉进纯黑箱 predictor 叙事 | 借语言和 discipline |
| `FormulationLAI` | `defend` | platform / formulation workflow / utility | 抢“更有用、更平台化”的叙事 | 防守 |
| `FormulationAI` | `defend` | AI-driven formulation decision-support | 抢 translational / decision-support 位置 | 防守 |
| active-learning / optimization systems | `defend` | 闭环优化 / assay planning | 抢“会指导实验”的位置 | 防守 |

## 3. 要正面对打的对象

### 3.1 NC / Bannigan

为什么一定要正面对打：

- 它是 reviewer 最熟悉的 few-shot release ML 路线
- 它代表外部默认问题定义：
  - `descriptors +/- early points -> release`
- 如果这条线不正面对齐，我们的项目很容易被误读成“更复杂但不必要”

我们的正确打法不是：

`我有更复杂的模型`

而是：

`我把 same task 提升成了 shared partially observed inference problem`

本地支撑证据：

- [summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)
- [theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)

### 3.2 explainable LAI few-shot / direct-release predictor

为什么要打：

- 它们更简单、更容易懂
- 更容易在审稿里被当成“够用了”
- 会削弱我们对 `middle object / family object` 的必要性叙事

正确打法：

- 不和它们比“有没有更简洁”
- 而是比：
  - harder OOD
  - shared object
  - beyond curve-fit targets

### 3.3 新的更大规模 PLGA supervised predictor

为什么要打：

- 会让我们被重新缩回 `PLGA-local benchmark`
- 容易把“统一药物释放”打回“一个材料体系的更强 predictor”

正确打法：

- 直接承认 PLGA predictor 不是 open slot
- 把 open slot 写成：
  - shared benchmark
  - posterior-family object
  - first non-PLGA bridge

## 4. 最值得借的对象

### 4.1 liposome IVR workflow

为什么值得借：

- 它帮我们把 non-PLGA bridge 做真实
- 它不是来和我们抢 shared stack 的
- 它对 assay summary、mechanism-appropriate packaging 很有用

最该借的不是模型，而是：

- assay discipline
- mechanism-local decoder discipline
- intake / packaging style

### 4.2 Scientific Data / AI-ready release data

为什么值得借：

- 这类工作决定以后 field 里什么算“标准化 release 数据”
- 如果我们不借，会一直像 bespoke pipeline

最该借的是：

- schema-first intake
- metadata-first federation
- benchmark packaging discipline

### 4.3 computational pharmaceutics / hybrid modeling

为什么值得借：

- 这类语言能帮我们摆脱“纯黑箱 predictor”
- 更适合解释：
  - mechanism-specific decoders
  - feasible-state family
  - release decision-support

一句话：

`该借的是 discipline，不是照抄一个别人的 predictor。`

## 5. 最需要防守的对象

### 5.1 FormulationLAI

危险不在 benchmark，而在叙事：

- 它容易看起来更像“真正可用的平台”
- 如果我们只讲 metric，它会比我们更像未来产品

所以对它的防守不是去模仿它，而是把我们的输出改写成：

- release decisions
- uncertainty-aware release windows
- assay value of next observation

### 5.2 FormulationAI

危险点：

- 抢 `AI + formulation decision-support` 的上位叙事

防守方式：

- 把我们的 scientific core 和 decision-support 语言接起来
- 不再只说“预测曲线”
- 要说：
  - duration
  - tail behavior
  - uncertainty
  - assay planning

### 5.3 active-learning / optimization systems

危险点：

- 它们可能模型没我们深，但看起来更会指导实验

防守方式：

- 不急着假装我们已经平台化
- 先把 `release intelligence -> release decision-support` 这条线写清楚

## 6. 现在哪些层我们已经领先？

当前最确定领先的是：

1. `L2 partial-observation benchmark`
2. `L3 shared posterior-family object`

直接数表：

- [stack_layer_advantage_summary.csv](D:/release-foundation/outputs/109_release_stack_occupancy_comparison/stack_layer_advantage_summary.csv)
  - `L2`: 对最强外部 `+0.50`
  - `L3`: 对最强外部 `+1.00`

这意味着：

`我们当前最硬的 moat 是 object/benchmark 层，不是 platform 层。`

## 7. 现在哪些层我们还容易被打？

当前最容易被外部压住的层是：

1. `L5 reporting / decision layer`
2. 简洁 benchmark narrative
3. completed prospective cross-mechanism validation

最直白地说：

- 我们比很多人更深
- 但还不比平台型路线更“有用感”

这就是为什么现在最该补的不是再加一个 benchmark cell，而是：

`route-consistent uncertainty-aware duration / shape`

## 8. 如果要一句一句地说打法

### 对 benchmark rival

`正面打。`

打的不是“我也会做 release regression”，而是：

- sparse early observation 下的 harder OOD forecasting
- feasible-state family
- cross-mechanism viability

### 对 borrowing target

`大胆借。`

借的是：

- schema
- workflow discipline
- assay packaging
- metadata federation

### 对 platform threat

`提前防。`

防的不是分数，而是：

- utility narrative
- translational framing
- decision-support position

## 9. 最后的作战原则

如果后面要持续推进，这张 battlecard 对应的原则就是：

1. `benchmark rival 要打`
2. `workflow / data discipline 要借`
3. `platform narrative 要防`
4. `不要再把自己缩成一个更强 PLGA predictor`
5. `继续把 open slot 写成 shared release-intelligence stack`

## 证据入口

- [drug_release_strategy_answer_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_strategy_answer_zh_2026-05-29.md)
- [drug_release_competitive_action_map_2026-05-29.md](D:/release-foundation/docs/drug_release_competitive_action_map_2026-05-29.md)
- [drug_release_stack_competitor_scorecard_2026-05-29.md](D:/release-foundation/docs/drug_release_stack_competitor_scorecard_2026-05-29.md)
- [drug_release_outside_in_strategy_memo_2026-05-29.md](D:/release-foundation/docs/drug_release_outside_in_strategy_memo_2026-05-29.md)
- [release_stack_occupancy_verdict_2026-05-29.md](D:/release-foundation/docs/release_stack_occupancy_verdict_2026-05-29.md)

