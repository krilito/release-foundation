---
date: 2026-06-03
language: zh
scope: plga_release_algorithm_decision
depends_on:
  - docs/algorithm_shortlist_for_plga_release_2026-06-03.md
  - docs/algorithm_repo_mapping_for_plga_release_2026-06-03.md
---

# 我们现在该选什么算法

这份不是文献综述。

这份只回答一个问题：

```text
按我们当前项目的真实痛点，下一步最该试什么，不该试什么。
```

## 先把问题说死

我们现在不是一个普通的“曲线拟合”项目。

从仓库现状看，真实问题是这四件事同时存在：

1. `Q(t)` 是累计释放，必须单调、有限、有饱和上界。
2. 数据是 `少曲线 + 少时间点 + 跨 DOI/source 分布漂移`。
3. 现有外部强基线不是神经网，而是 `Direct LGBM / RF / tree` 这一类。
4. 我们真正没解决的不是中位数 R²，而是 `cross-DOI bad tail`。

本地证据：

- [ARCHITECTURE.md](D:/release-foundation/ARCHITECTURE.md:8)
- [SPRINT1_STATUS_NOTE.md](D:/release-foundation/SPRINT1_STATUS_NOTE.md:203)
- [DECISIONS.md](D:/release-foundation/DECISIONS.md:1624)
- [DECISIONS.md](D:/release-foundation/DECISIONS.md:1627)
- [DECISIONS.md](D:/release-foundation/DECISIONS.md:1928)

## 先分清两类任务

### 任务 A：descriptor-only / direct-Q

形式：

```text
配方特征 x (+ 可选 t_query) -> Q(t)
```

这类任务的核心矛盾是：

- 如何在不作弊的前提下做 `cross-DOI`
- 如何保证输出曲线合法
- 如何在不引入重型模型的前提下给 tree baseline 一个公平对手

### 任务 B：few-shot / sparse-prefix forecasting

形式：

```text
配方特征 x + 少量早期观测点 {(t_i, Q_i)} -> 后续整条曲线
```

这类任务的核心矛盾是：

- 前缀观测稀疏且时间不规则
- 模型要处理“知道一点，但不知道全部”的信息结构
- 需要把不确定性当真，不是只给一个点预测

这两类任务不要混着打。

这是我们过去一个隐性问题：

- direct-Q 强项和 prefix forecasting 强项不是同一个模型族
- 你用错赛道，算法看起来就像“没用”

## 现在最该试的算法，按优先级排

## 1. `Deep Lattice` 或 `UMNN`

### 这是干什么的

这是给 `direct-Q` 用的，不是给 world model 用的。

作用是：

- 原生保证 `Q(t)` 对 `t` 单调
- 让 cumulative release 曲线从模型内部就是合法的
- 替掉现在那种“先预测，再拿 isotonic/PAVA 补救”的做法

### 为什么适合我们

因为我们现在最便宜、最合理、最 reviewer-friendly 的升级，
不是先上更复杂的潜变量系统，而是先给 `DirectQ` 一条
**真正受约束的强基线**。

### 它接到哪条线

- 首选：[scripts/72_canonical_benchmark_v2.py](D:/release-foundation/scripts/72_canonical_benchmark_v2.py:1)
- 次选：[scripts/80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py:1)

### 我对它的判断

如果下一轮只允许做一个新 baseline，就做这个。

## 2. `GP-ConvCNP` / `CNP` / `Sequential NP`

### 这是干什么的

这是给 `few-shot sparse-prefix forecasting` 用的。

作用是：

- 看到少量前缀点，就去预测未来整条函数
- 比 kNN / 普通 RNN 更符合“少上下文点推完整函数”的任务定义
- 天生更适合不规则采样和不确定性输出

### 为什么适合我们

因为你们仓库已经越来越清楚：

- 真正 deployable 的场景不是纯 descriptor-only
- 而是“给一点早期释放，再继续预测”

这正是 CNP 家族该上的场景。

### 它接到哪条线

- 首选：[scripts/23_prefix_world_model_benchmark.py](D:/release-foundation/scripts/23_prefix_world_model_benchmark.py:1)
- 统一 benchmark 层：[scripts/80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py:1)

### 我对它的判断

如果你要打的主故事是 `sparse-prefix OOD forecasting`，这是最像正解的一条。

## 3. `Neural CDE`

### 这是干什么的

这是一个 **prefix encoder 替换件**，不是独立 benchmark 大类。

作用是：

- 处理不规则时间点
- 比 GRU/LSTM 更自然地编码连续时间观测

### 为什么它不是第一优先级

因为它解决的是“怎么编码不规则前缀”，
但我们现在更大的问题经常是：

- target object 选错了没有
- benchmark lane 站对了没有
- direct-Q 有没有一个公平的强约束 baseline

这些没理清之前，先上 CDE 容易变成技术炫技。

### 它接到哪条线

- [scripts/26_iterative_world_model.py](D:/release-foundation/scripts/26_iterative_world_model.py:1)
- 或者更干净地接到 [docs/refactor_27_npe.md](D:/release-foundation/docs/refactor_27_npe.md:1) 这条 prefix-posterior 线

### 我对它的判断

它值得做，但应该排在 CNP 和 monotonic direct-Q 之后。

## 4. grouped sparse FDA / functional mixed-effects

### 这是干什么的

这是统计学防守 baseline，不是未来主干。

作用是：

- 正面回应“你有没有认真比较 sparse functional statistics”
- 让 DOI / study grouping 在模型里显式存在

### 为什么它值得有

因为我们这个数据几何很像它擅长的地方：

- 小样本
- 稀疏曲线
- source grouping

### 它接到哪条线

- [scripts/38b_canonical_functional_baselines.py](D:/release-foundation/scripts/38b_canonical_functional_baselines.py:1)

### 我对它的判断

它不是最强模型，但它是很强的 reviewer defense。

## 5. `UDE` / residual neural ODE

### 这是干什么的

这是最像“机理 + ML 升级”的正统路线。

作用是：

- 保留 PLGA ODE 主体
- 只让网络去学 parametric ODE 没覆盖到的残差动力学

### 为什么现在不该先做它

因为它太贵了。

不是代码贵，是科学问题贵：

- 识别性风险更高
- 校准要求更高
- 一旦没打过 baseline，你很难解释到底是架构错、目标错还是数据不支持

### 它接到哪条线

- [plan_27d_residual_neural_ode.md](D:/release-foundation/docs/plan_27d_residual_neural_ode.md:1)

### 我对它的判断

这是中期方法主线，不是下一周最该做的事。

## 哪些东西不该再继续手搓

## 1. `kNN` 做主解

它可以当 sanity check，但不像正解。

因为我们的问题不是“找相似曲线”，而是：

- 少量信息下的函数外推
- 跨 source 分布漂移
- 合法形状约束

## 2. 再造一个 `Weibull` 变体

Weibull 族有它的地位，但继续把它当“新方法候选”往前推，
研究价值已经不高了。

更合理的角色是：

- shape-family audit
- 低维解释基线
- 诊断工具

## 3. 把 `isotonic/PAVA` 当创新

它应该被降级成：

- 免费 shape repair baseline
- sanity layer

不是方法主角。

## 4. 把 `world model` 当默认未来方向

仓库很多文档其实已经在提醒这件事：

- world model 说法太重
- prefix forecasting / posterior family / release intelligence 才是更稳的表述

所以不要因为模型看起来高级，就默认它更对。

## 我给你的实验顺序

## 第一顺位：先做这个

在 `72` 那条线加一个：

```text
monotonic direct-Q baseline
```

候选实现：

- Deep Lattice
- 或 UMNN

目标不是马上干翻一切，而是回答：

```text
一个原生单调的 direct-Q 模型，能不能在公平 benchmark 上
比现有 unconstrained direct baseline 更像样？
```

## 第二顺位：再做这个

在 `23/80` 那条线做一个：

```text
GP-ConvCNP few-shot prefix forecaster
```

目标是回答：

```text
当任务明确定义为 sparse-prefix -> future curve 时，
CNP 家族是不是比当前 heuristic/world-model 路线更自然？
```

## 第三顺位：补这个防守项

在 `38b` 那条线补一个：

```text
grouped sparse FDA / functional mixed-effects baseline
```

目标是防 reviewer，不是当主角。

## 第四顺位：最后再上这个

走 `27d` 的 residual-UDE 路线。

前提是前面三件事做完后，仍然存在一个明确的、
值得花高成本去补的 `cross-DOI tail gap`。

## 如果只允许我给一句话

那就是：

```text
先别急着上更重的模型。
先在 direct benchmark 上做一个原生单调的强 baseline，
再在 sparse-prefix benchmark 上试一个 CNP 家族模型。
```

这两个做完，很多路线会自动变清楚。

## 浏览器说明

我尝试过用 in-app Browser 走 `@chrome` 这条路，但这台机器上的
Browser runtime 在初始化阶段连续两次直接退出，所以这轮结论不是
来自 live browser session，而是来自：

- 已核验的网页检索
- 公开论文页面
- 仓库现有证据链

这不影响算法判断本身，但说明当前 `@chrome` 工作流在这台环境里不稳定。
