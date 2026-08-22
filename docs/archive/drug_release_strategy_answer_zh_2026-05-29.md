# 药物释放统一战略回答（中文）

日期：`2026-05-29`

目的：

```text
直接回答最初那几个问题：
1. 现在外面的 ML + 药物释放主要都在怎么做？
2. 我们到底能不能尝试统一整个药物释放？
3. 胆子可以大到哪里，哪里又不能乱吹？
4. 现在的竞品是谁，哪些路线值得借鉴？
5. 下一步最值钱的动作是什么？
```

## 1. 一句话总判断

`可以胆子大，但要大胆在“shared release-intelligence stack”，而不是大胆吹“一个 universal solved model”。`

更具体一点：

- 现在外面主流还在做 `descriptor ML`、`few-shot early-point forecasting`、单机制 workflow、平台化 formulation AI。
- 我们已经不只是一个 `PLGA predictor`，而是已经做出了：
  - shared benchmark contract
  - first non-PLGA liposome bridge
  - shared posterior-family object
  - mechanism-aware target registry
  - calibrated-family panel
- 所以现在最诚实也最大胆的定位是：
  - `unified drug-release intelligence`
  - 或者更硬一点：`shared release-intelligence stack`

## 2. 外面现在到底在怎么做？

当前外部赛道基本分成 5 类：

1. `descriptor-only supervised prediction`
输入配方/材料描述符，直接预测 release curve 或 summary 指标。

2. `descriptor + sparse early-point forecasting`
给少量早期释放点，再预测后面的曲线。这类最接近 `NC / Bannigan` 路线。

3. `mechanism-local kinetic / assay workflows`
不追求一个统一模型，而是在某一机制里做 kinetic fit、IVR workflow、机制摘要。

4. `platform / optimization / formulation-AI systems`
更强调 workflow、决策支持、配方优化，而不是单个 benchmark 分数。

5. `data / schema / benchmark infrastructure`
这类工作不一定直接拼模型效果，但会定义以后什么叫“标准输入”和“标准 benchmark”。

最关键的 outside-in 结论是：

`外面现在并没有谁真正把 drug release 组织成一个 shared partially observed inference stack。`

证据入口：

- [drug_release_master_positioning_dossier_2026-05-29.md](D:/release-foundation/docs/drug_release_master_positioning_dossier_2026-05-29.md)
- [drug_release_outside_in_strategy_memo_2026-05-29.md](D:/release-foundation/docs/drug_release_outside_in_strategy_memo_2026-05-29.md)
- [release_stack_occupancy_verdict_2026-05-29.md](D:/release-foundation/docs/release_stack_occupancy_verdict_2026-05-29.md)
- [drug_release_external_paper_journal_inventory_2026-05-29.md](D:/release-foundation/docs/drug_release_external_paper_journal_inventory_2026-05-29.md)
- [drug_release_external_refresh_2026-05-29.md](D:/release-foundation/docs/drug_release_external_refresh_2026-05-29.md)

## 3. 我们到底能不能统一整个药物释放？

### 能统一的层

现在证据支持统一的是下面这几层：

1. `shared schema`
2. `shared partially observed benchmark`
3. `shared posterior-family object`
4. `mechanism-specific decoders`
5. `shared target / reporting semantics`

这就是为什么现在更对的说法不是：

`one universal mechanistic model`

而是：

`shared release-intelligence stack`

### 不能硬统一的层

现在绝对不能写成已经统一的是：

1. `所有机制共用一个 raw physical theta`
2. `一个 universal t80 可以当所有机制的默认 timing target`
3. `一个模型已经 solved cross-mechanism release prediction`

证据层已经把这个边界钉得很死：

- `L2` 和 `L3` 是我们最强的统一层
- `L5 reporting / decision` 还只能保守写
- 尤其深阈值 timing target 还不成熟
- 同时，route-consistent duration/shape uncertainty 现在已经推进到当前 full `9-cell calibrated-family cell panel`
  - `liposome` 三个 cell 加 `internal181` 三个 cell 已经 full matched-curve
  - 只剩 `cross321` 三个 cell 还在 within-cell sampled
  这一层，但还没到 benchmark-wide

对应数表：

- [stack_layer_advantage_summary.csv](D:/release-foundation/outputs/109_release_stack_occupancy_comparison/stack_layer_advantage_summary.csv)
  - `L2`: 我们 `1.0`，最强外部 `0.5`
  - `L3`: 我们 `1.0`，最强外部 `0.0`
  - `L5`: 我们 `0.75`，最强外部 `1.0`

这就是当前 moat 的核心：

`不是最好单模型，而是 L2-L3 的 stack occupancy。`

而在最难的 duration/UQ 这条线上，最新状态也更清楚了：

- shape-aware uncertainty 已经能跨当前 full `9-cell` cell panel 汇总，但 coverage 结构是 mixed，不是所有 cell 都 full-curve
- deep-threshold timing 仍然是主要瓶颈

## 4. 胆子到底可以大到哪里？

### 现在可以大胆写的

1. `Sparse early release is the dominant OOD signal.`
2. `Mechanism-routed middle objects outperform matched direct-Q routes on the audited PLGA core.`
3. `The project already supports a shared benchmark, a shared posterior-family object, and a first non-PLGA bridge.`
4. `The most defensible unification level is a shared release-intelligence stack with mechanism-specific decoders.`

### 现在不能写的

1. `universal solved drug-release model`
2. `cross-mechanism validation complete`
3. `uncertainty solved across mechanisms`
4. `universal deep-threshold timing target`

最短版本的胆量边界是：

`我们可以大胆占“统一 release-intelligence stack”这个位置，但还不能大胆吹“已经统一整个药物释放模型”。`

## 5. 我们现在最强的证据到底是什么？

### PLGA audited core

当前最硬的基础证据：

- formulation-only 的 OOD best median `R²` 只有 `0.525–0.672`
- early-only 的 OOD best median `R²` 是 `0.907–0.949`
- theta-minus-direct 的 OOD 增益范围是 `+0.002` 到 `+0.121`

这说明：

1. `纯配方` 在 harder OOD 下不够稳
2. `早期释放观测` 才是真正主信号
3. `mechanism-routed middle object` 不是装饰

来源：

- [summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)

### Liposome first non-PLGA bridge

liposome bridge 已经证明我们不是 PLGA-only：

- `group_by_API` pooled `R²`: mechanism `0.903` vs direct `0.864`
- `group_by_release_method` pooled `R²`: mechanism `0.903` vs direct `0.457`

但它也顺手证明了一件更诚实的事：

`没有任何一条路线现在能在 curve / timing / shape 上全面通吃。`

### UQ 和 prospective discipline

我们已经有：

- `PLGA + liposome` 的 calibrated-family panel
- chitosan preregistered lock

所以现在已经不是纯 retrospective point prediction paper 了。

## 6. 现在最大的短板是什么？

最大的短板已经不是“没有 UQ”或者“没有 non-PLGA”了，而是：

`route-consistent uncertainty-aware duration / shape reporting`

也就是说，我们现在已经分别有：

- curve panel
- timing / shape panel
- posterior-family object
- calibrated-family panel

但这些层还没有在同一路线上完全连起来。

### 更细一点：timing 和 shape 的成熟度不一样

当前 evidence 很明确：

`shape-aware targets 比 deep-threshold timing targets 成熟得多。`

shared timing target 的数表是：

- [threshold_target_scorecard.csv](D:/release-foundation/outputs/104_release_threshold_target_summary/threshold_target_scorecard.csv)
  - `t50`: truth `0.889`, closed `0.421`, covered `0.272`
  - `t60`: truth `0.819`, closed `0.201`, covered `0.126`
  - `t70`: truth `0.819`, closed `0.141`, covered `0.082`
  - `t80`: truth `0.819`, closed `0.064`, covered `0.056`

这意味着：

- `t50` 现在是 least-bad shared timing anchor
- `t80` 绝对不该再装作 universal default

shape target 的 registry 也已经比较明确：

- [registry_summary.csv](D:/release-foundation/outputs/106_release_target_registry/registry_summary.csv)
  - `plga`: 默认更像 `post_window + residual_tail + tail_auc`
  - `liposome`: 默认更像 `burst + post_window + residual_tail + tail_auc`
  - `chitosan`: reveal 前全部还是 placeholder

一句最短的话：

`统一 stack 现在最强的是 shape-aware target layer，不是 deep-threshold timing layer。`

## 7. 竞品是谁？

### 真正要正面对打的 benchmark rival

这些是 reviewer 最可能默认拿来比的：

1. `NC / Bannigan`
2. explainable LAI few-shot systems
3. 新的更大规模 PLGA supervised predictors

这部分要 `fight`，因为它们最像“直接 release predictor”。

### 真正危险的 platform threat

这些不一定和我们拼同一个 benchmark，但会抢“更有用”的叙事：

1. `FormulationLAI`
2. `FormulationAI`
3. active-learning / optimization systems

这部分要 `defend`，因为如果我们只会说 “prediction 更准”，它们仍然会看起来更像未来平台。

## 8. 哪些东西值得借？

最值得借的不是某个模型，而是它们各自占住的层：

1. 从 `NC / Bannigan` 借：
   - `few-shot / zero-shot` 叙事
   - prefix forecasting 的任务语言

2. 从 liposome workflow 和机制摘要类工作借：
   - `mechanism-local decoder` 的 discipline
   - non-PLGA intake / assay packaging

3. 从 Scientific Data / AI-ready data 借：
   - schema-first discipline
   - metadata/benchmark federation

4. 从 computational pharmaceutics / hybrid modeling 借：
   - “不是纯黑箱 predictor” 的科学语言

一句话：

`该借的是 schema、workflow、mechanism discipline，不是退回纯 direct regression。`

## 9. 如果现在要定一个最清楚的战略定位

### 最推荐的定位

`shared partially observed drug-release intelligence with mechanism-specific decoders`

### 中文直译版

`基于机制特异解码器的、共享的、部分观测药物释放智能框架`

### 最适合内部说的版本

`我们不是在做一个更强的药物释放回归器，而是在做 shared release-intelligence stack。`

### 最不该说的版本

`我们已经做成 universal drug-release foundation model 了。`

## 10. 下一步最值钱的动作是什么？

如果按“把项目做大、做真”排序，最值钱的下一步不是再多加一个 PLGA cell，而是：

1. `把 uncertainty-aware duration / shape 真正做成 route-consistent layer`
2. `等待并利用 chitosan reveal 做 credibility jump`
3. `再补一个比 liposome 更强的 non-PLGA bridge`

如果按 paper 和项目战略的共同收益来排，优先级我会写成：

1. `route-consistent uncertainty-aware duration / shape`
2. `completed chitosan reveal`
3. `further cross-mechanism evidence beyond liposome`

## 11. 最后一句最硬的话

`现在最值得占的位置，不是“最好单模型”，也不是“已经统一整个药物释放物理学”，而是“当前最完整的 shared release-intelligence stack”。`

## 证据入口

- [drug_release_master_positioning_dossier_2026-05-29.md](D:/release-foundation/docs/drug_release_master_positioning_dossier_2026-05-29.md)
- [drug_release_executive_brief_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_executive_brief_zh_2026-05-29.md)
- [drug_release_battlecard_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_battlecard_zh_2026-05-29.md)
- [drug_release_outside_in_strategy_memo_2026-05-29.md](D:/release-foundation/docs/drug_release_outside_in_strategy_memo_2026-05-29.md)
- [release_stack_occupancy_verdict_2026-05-29.md](D:/release-foundation/docs/release_stack_occupancy_verdict_2026-05-29.md)
- [release_target_registry_verdict_2026-05-29.md](D:/release-foundation/docs/release_target_registry_verdict_2026-05-29.md)
- [release_shape_target_contract_2026-05-29.md](D:/release-foundation/docs/release_shape_target_contract_2026-05-29.md)
- [summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)
- [stack_layer_advantage_summary.csv](D:/release-foundation/outputs/109_release_stack_occupancy_comparison/stack_layer_advantage_summary.csv)
- [threshold_target_scorecard.csv](D:/release-foundation/outputs/104_release_threshold_target_summary/threshold_target_scorecard.csv)
- [registry_summary.csv](D:/release-foundation/outputs/106_release_target_registry/registry_summary.csv)
