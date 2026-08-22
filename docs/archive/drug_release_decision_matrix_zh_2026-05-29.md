# 药物释放外部路线决策矩阵

日期：`2026-05-29`

目的：

```text
把外部主要对象压成一张“现在该怎么处理”的决策表：
1. 它对我们的压力类型是什么；
2. 当前压力强不强；
3. 我们应该打、借、还是防；
4. 为什么现在要这么处理。
```

这份文档比 battlecard 更短，也比 executive brief 更具体。  
如果需要快速决定“接下来把精力投到谁身上”，优先看这张表。

## 决策矩阵

| 对象 | 类别 | 当前压力 | 建议动作 | 现在为什么这样处理 |
| --- | --- | --- | --- | --- |
| `NC / Bannigan` | benchmark rival | 高 | `fight now` | 这是最直接的 benchmark 对手，抢的是 sparse-prefix 释放预测的正面话语权。 |
| explainable LAI few-shot | benchmark rival | 中高 | `fight now` | 它和我们在“早期观测 -> 后续释放”这一层高度相邻，会挤压我们的方法新意。 |
| 新的大规模 PLGA supervised predictor | benchmark rival | 中高 | `fight now` | 它会削弱“我们在 PLGA 上更懂 release”这件事，所以要用 audited core 和 middle-object 证据守住。 |
| liposome IVR workflow | infrastructure / workflow | 中 | `borrow now` | 这是我们最有价值的非 PLGA 借鉴对象，适合用来证明 shared interface 不是只适用于 PLGA。 |
| Scientific Data / AI-ready release data | infrastructure | 中 | `borrow now` | 它们不直接跟我们抢模型位置，但会定义别人心中的“标准数据组织方式”。 |
| computational pharmaceutics / hybrid modeling | scientific framing | 中 | `borrow now` | 这条线能给我们更稳的科学语言，避免把项目讲成普通黑箱回归。 |
| `FormulationLAI` | platform threat | 高 | `defend now` | 它抢的是“更有用的配方设计平台”叙事，不是单一 benchmark 分数。 |
| `FormulationAI` | platform threat | 高 | `defend now` | 它在 workflow / decision-support 语言上更强，会冲掉我们为什么重要这层叙事。 |
| active-learning / optimization systems | platform threat | 中高 | `defend selectively` | 这类系统短期不一定直接压我们 benchmark，但会让我们显得“不够像一个真正可用的平台”。 |

## 读表原则

### `fight now`

满足两个条件就该直接打：

- 它和我们抢同一层 scientific credit
- 我们手里已经有本地证据能正面对比

当前最典型的是：

- `NC / Bannigan`
- explainable LAI few-shot
- 更大规模的 PLGA direct predictor

### `borrow now`

这类对象不必正面对抗，重点是借它们的：

- schema discipline
- workflow packaging
- scientific language
- non-PLGA assay organization

### `defend now`

这类对象最危险的地方不是分数，而是：

- utility narrative
- translational framing
- decision-support position

所以这里的防守重点不是“证明我们也会做 optimization”，而是：

`说明我们占的是 shared release-intelligence stack，而不是单一回归器。`

## 一句话结论

`现在最该正面打的是 benchmark rival，最该借的是 workflow/schema discipline，最该防的是 platform narrative。`

## 快速入口

- 中文 1 页 brief：
  [drug_release_executive_brief_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_executive_brief_zh_2026-05-29.md)
- 中文总回答：
  [drug_release_strategy_answer_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_strategy_answer_zh_2026-05-29.md)
- 中文 battlecard：
  [drug_release_battlecard_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_battlecard_zh_2026-05-29.md)
- 总入口：
  [drug_release_master_positioning_dossier_2026-05-29.md](D:/release-foundation/docs/drug_release_master_positioning_dossier_2026-05-29.md)
