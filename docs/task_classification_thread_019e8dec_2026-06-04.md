---
date: 2026-06-04
thread_id: 019e8dec-c2d4-7a00-b807-dc2b7865d68a
scope: completed_task_classification
depends_on:
  - docs/algorithm_shortlist_for_plga_release_2026-06-03.md
  - docs/algorithm_repo_mapping_for_plga_release_2026-06-03.md
  - docs/algorithm_decision_note_zh_2026-06-03.md
  - docs/algorithm_implementation_options_2026-06-03.md
---

# 线程 `019e8dec-c2d4-7a00-b807-dc2b7865d68a` 已完成任务分类

这份文档只做一件事：

```text
把该线程已经完成的工作按功能分类，避免后续把
"文献对标"、"仓库落点"、"决策建议"、"实现可行性"
混成一团。
```

## 一、研究对标类

这部分回答：

```text
我们当前问题在外部文献里已经有哪些成熟坐标系？
哪些方向是现成轮子，哪些只是我们自己在重复发明？
```

已完成内容：

- 核验 `Bannigan et al., Nature Communications 2023`，确认
  `PLGA release prediction` 的公开 benchmark 线已经存在。
- 核验 `Kapoor & Narayanan, Patterns 2023`，确认
  `non-independent train/test split` 可以直接挂到标准 leakage 框架。
- 核验单调模型线：`UMNN`、`Deep Lattice`、`Monotonic GP`。
- 核验 few-shot 曲线模型线：`CNP / GP-ConvCNP`。
- 核验机理 + ML 升级线：`UDE / Neural ODE`。
- 核验 `Weibull` 谱系，确认它属于经典溶出模型，不是新的方法学方向。

对应文档：

- [algorithm_shortlist_for_plga_release_2026-06-03.md](D:/release-foundation/docs/algorithm_shortlist_for_plga_release_2026-06-03.md)

## 二、仓库映射类

这部分回答：

```text
这些算法如果真要做，应该落到仓库哪条脚本线，
而不是停留在 paper name 层面。
```

已完成内容：

- 把 `monotonic direct-Q baseline` 映射到
  [scripts/72_canonical_benchmark_v2.py](D:/release-foundation/scripts/72_canonical_benchmark_v2.py:1)
- 把 `few-shot sparse-prefix forecasting` 映射到
  [scripts/23_prefix_world_model_benchmark.py](D:/release-foundation/scripts/23_prefix_world_model_benchmark.py:1)
  和
  [scripts/80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py:1)
- 把 `grouped sparse FDA baseline` 映射到
  [scripts/38b_canonical_functional_baselines.py](D:/release-foundation/scripts/38b_canonical_functional_baselines.py:1)
- 把 `residual-UDE` 识别为中期路线，对应
  [plan_27d_residual_neural_ode.md](D:/release-foundation/docs/plan_27d_residual_neural_ode.md:1)

对应文档：

- [algorithm_repo_mapping_for_plga_release_2026-06-03.md](D:/release-foundation/docs/algorithm_repo_mapping_for_plga_release_2026-06-03.md)

## 三、决策收束类

这部分回答：

```text
下一步最该试什么，不该再重投入什么？
```

已完成内容：

- 明确区分两类任务：
  `descriptor-only / direct-Q`
  与
  `few-shot / sparse-prefix forecasting`
- 给出实验优先级：
  1. 先做 `72` 线上的 `UMNN / Deep Lattice` monotonic baseline
  2. 再做 `23/80` 线上的 `CNP / GP-ConvCNP`
  3. 再补 `38b` 上的 serious grouped sparse FDA baseline
  4. 最后才进入 `27d residual-UDE`
- 明确降级以下方向，不再当主创新重投入：
  `kNN`
  `Weibull` 新变体
  把 `isotonic/PAVA` 当方法创新

对应文档：

- [algorithm_decision_note_zh_2026-06-03.md](D:/release-foundation/docs/algorithm_decision_note_zh_2026-06-03.md)

## 四、实现可行性类

这部分回答：

```text
哪些方向今天就能开干，哪些虽然科学上合理，
但在当前栈里并不省事？
```

已完成内容：

- 核对当前依赖栈，确认仓库是 `PyTorch + sbi + torchdiffeq`。
- 确认当前没有 TensorFlow 依赖，因此：
  - `UMNN` 更贴当前工程栈
  - `TensorFlow Lattice` 科学上合适，但生态不贴当前仓库
- 确认 `torchcde` 更适合作为 irregular prefix encoder 路线的工程入口。
- 确认 `CNP / GP-ConvCNP` 科学上对题，但不是最省事的第一枪。
- 确认 `UDE` 生态成熟，但对当前仓库来说更像中期计划。

对应文档：

- [algorithm_implementation_options_2026-06-03.md](D:/release-foundation/docs/algorithm_implementation_options_2026-06-03.md)

## 五、已识别阻塞类

这部分回答：

```text
线程里哪些事不是没做，而是被外部环境卡住了？
```

已完成判断：

- `@chrome / in-app browser` 路线尝试过多次，但 browser runtime
  初始化即退出，属于外部执行面阻塞。
- 因此该线程的结论虽然完成了，但证据获取路径最终是：
  `已核验网页来源 + 仓库本地证据`
  而不是 live browser session。

这类阻塞不影响当前四份文档的研究判断，但影响后续
“直接在应用内浏览器持续追踪网页”的工作流稳定性。

## 六、最终交付物清单

- [algorithm_shortlist_for_plga_release_2026-06-03.md](D:/release-foundation/docs/algorithm_shortlist_for_plga_release_2026-06-03.md)
- [algorithm_repo_mapping_for_plga_release_2026-06-03.md](D:/release-foundation/docs/algorithm_repo_mapping_for_plga_release_2026-06-03.md)
- [algorithm_decision_note_zh_2026-06-03.md](D:/release-foundation/docs/algorithm_decision_note_zh_2026-06-03.md)
- [algorithm_implementation_options_2026-06-03.md](D:/release-foundation/docs/algorithm_implementation_options_2026-06-03.md)
- [task_classification_thread_019e8dec_2026-06-04.md](D:/release-foundation/docs/task_classification_thread_019e8dec_2026-06-04.md)

## 七、一句话结论

这个线程完成的不是“再发明一个新模型”，而是把：

- 外部文献坐标
- 仓库脚本落点
- 实验优先级
- 当前实现可行性
- 外部阻塞边界

一次性分开、说清、固化进仓库。
