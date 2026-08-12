# 三频架构实验路线图 — 每步的目标与目的

> 核心理念：每一步都在验证一个可证伪的假设。不仅是在"搭系统"，而是在逐层证明三频架构的每一层收益真实存在。

---

## 假设清单（按验证顺序）

| 编号 | 假设 | 如果成立意味着什么 | 证伪代价 |
|:---:|------|-------------------|:---:|
| H1 | Cosmos 3 AR 塔在 NPU 上能正常推理 | 三频架构的 AR 层存在，软件基础打通 | 0 天（直接放弃这条路） |
| H2 | AR 的语义条件信号优于 T5/FastWAM | 三频 > 双频（FastWAM），核心优势成立 | 3 天（但 FastWAM 本身也可用） |
| H3 | Prefill cache + Action-only 推理能达到实时延迟 | 多频推理的时延收益成立，可部署 | 1-2 周（工程优化可补救） |
| H4 | AR 闭环能正确判断执行偏差 | Test-Time Scaling 的前提成立 | 1-2 周（可改进 prompt） |
| H5 | RL 训练的样本效率优于传统 baseline | 自我提升飞轮成立，第三条腿站稳 | 3-6 周（可降级为纯 BC 微调） |

---

## Phase 0：AR 塔独立工作（H1 验证）

### 实验 A：AR 塔 NPU 推理 → VLM 能力基准

**目的**：验证 H1——拆出的 AR 塔能独立工作。

**假设备择**：如果 AR 塔在 NPU 上跑不通，三频架构不成立。需要回退到用独立的 Qwen3-VL 或 FastWAM 的 T5 作为文本编码器。

**方法**：
```
Cosmos3-Nano checkpoint → 加载 → .to("npu") → AR forward (text+image→text)

测试用例（直接用 Cosmos 3 cookbook 的 reasoner 示例）:
  1. 图像描述 (Image Caption)
  2. 具身推理 (Embodied Reasoning - "下一步做什么?")
  3. 物理合理性 (Physical Plausibility)
  4. 2D 定位 (2D Grounding)
```

**成功标准**：
- AR forward 不报错（所有 op NPU 支持）
- 输出文本语义正确（与 CUDA 参考一致，或人工评估合理）
- 延迟 < 2 秒（prefill ~0.5s + decode ~1.5s, 100 tokens）

**产出物**：AR NPU 推理的延迟基准、显存占用、输出质量评估。

---

## Phase 1：AR → Action 条件信号（H2 验证）

### 实验 B：纯文本→Action（跳过 Video 中频）

**目的**：验证 H2——AR 的语义理解能否产出比 T5 更好的 Action 条件信号。

**这是三频架构最核心的假设。如果 H2 不成立，三频相比 FastWAM 没有本质优势。**

**方法**：
```
控制变量实验:
  Variant A (FastWAM baseline):  T5 文本编码器 → Action DiT
  Variant B (三频验证):           Cosmos 3 AR 塔 → 冻结 → text embedding → Action DiT

两组使用完全相同的 Action DiT 架构和训练数据（LIBERO-10 或 DROID 子集）。
只换文本编码器。
```

**对比指标**：
| 指标 | Variant A (T5) | Variant B (AR) | 判断 |
|------|:---:|:---:|------|
| Action MSE | 基线 | 预期更低 | AR 条件信号更精准 |
| 任务成功率 | 基线 | 预期更高 | AR 理解指令更深 |
| 未见指令泛化 | 基线 | **关键指标** | 物理推理能力 → 零样本迁移 |
| 训练收敛速度 | 基线 | 预期更快 | VLM 初始化 vs T5 冷启动 |

**成功标准**：
- Variant B 在至少 2/4 指标上优于 Variant A
- 特别是在"未见指令泛化"上有明显提升（这是 AR 物理推理和语义理解的核心价值）

**如果 H2 不成立（B ≤ A）**：说明 AR 的额外知识没有被有效利用。可能原因：
- AR embedding 需要在机器人数据上微调（加 LoRA adapter）
- AR 的输出维度与 Action DiT 的输入维度需要更好的 projection 对齐
- 需要更多的训练数据来"唤醒" AR 的物理知识

**产出物**：H2 成立的第一个 hard evidence（或 H2 不成立的明确归因）。

---

## Phase 2：多频时延验证（H3 验证）

### 实验 C：延迟基准测试

**目的**：验证 H3——多频推理能否达到实时控制延迟。

**方法**：
```
三频推理延迟 = Prefill AR K/V (一次) + N × Action DiT 去噪步

对比:
  原版 Cosmos 3 Policy (完整 16B, 50 步):     ~10-30s  ❌ 不可部署
  原版 Cosmos 3 Policy (蒸馏 4 步):            ~1-3s    ⚠️ 勉强
  FastWAM infer_action (双频, ~300M, N步):      ~0.5-1s  ⚠️ 临界
  三频 Action-only (AR prefill + ~300M, N步):   ~50-200ms ✅ 实时
```

**成功标准**：
- Action chunk 推理延迟 < 200ms → 等效控制频率 > 30Hz（chunk_size=32, execute 8）
- AR prefill 延迟 < 2s → 摊到 N 次 chunk 后，对于分钟级任务的 overhead < 3%
- 如果蒸馏到 4 步，延迟 < 50ms

**产出物**：多频推理的延迟数字，可以直接与 FastWAM 论文和 Cosmos 3 推理基准对比。

---

## Phase 3：闭环 Test-Time Scaling（H4 验证）

### 实验 D：AR 偏差检测精度

**目的**：验证 H4——AR 作为闭环监督者的判断精度。**这是 Test-Time Scaling 和 RL 的共同前提。**

**方法**：
```
构造已知偏差的测试轨迹:
  正常轨迹: ground truth 机器人数据
  偏差轨迹: 在正常轨迹上注入已知误差
    - Gripper 偏移 (3°, 5°, 10°)
    - 物体滑落 (cup 从 gripper 中消失)
    - 轨迹偏离 (偏离预期路径 2cm, 5cm)
    - 碰撞 (手臂碰到桌子)

让 AR 对每条轨迹做监督判断:
  1. 偏差检测: "有没有问题?" (二分类)
  2. 偏差分类: "什么问题?" (多分类)
  3. 严重度评估: "是否需要重规划?" (三分类: continue/adjust/replan)
  4. 修正建议: "具体怎么改?" (自由文本或结构化 JSON)
```

**成功标准**：
| 指标 | 目标 | 说明 |
|------|:---:|------|
| 偏差检测 Precision | > 85% | 正常时不要误报 |
| 偏差检测 Recall | > 90% | 有问题时不要漏检 |
| 偏差分类 Accuracy | > 75% | 能区分"gripper错位" vs "物体滑落" |
| 严重度判断 Accuracy | > 80% | 区分"小调整" vs "重规划" |

**如果 H4 不达标**：改进 AR 的 prompt 设计（给具体的判断标准和示例），或加一个轻量的外挂 classifier 做 complementary。

**产出物**：AR 监督能力的量化基准，决定了后续 Test-Time Scaling 和 RL 的可靠程度。

---

## Phase 4：RL 样本效率（H5 验证）

### 实验 E：RL vs 纯 BC 微调

**目的**：验证 H5——AR 引导的 RL 能否比纯 Behavior Cloning 更样本高效。

**方法**：
```
Baseline (纯 BC):
  用 ground truth action 做 Behavioral Cloning 微调

RL (AR 引导):
  Mode 1 (结果级): AR 对每条轨迹的 episode-level critique
    → 成功片段做 BC，失败片段做 hindsight relabeling
  
  Mode 2 (步骤级): AR + Video imagination 多候选评分
    → DPO 偏好优化

对比指标:
  - 达到相同成功率所需的数据量 (sample efficiency)
  - 在相同数据量下的成功率 (performance)
  - 泛化到未见任务的能力 (zero-shot transfer)
```

**成功标准**：
- Mode 1 的样本效率 > 纯 BC（用更少数据达到相同成功率）
- Mode 2 ≥ Mode 1（Video imagination 提供了额外的增益）
- 如果 Mode 2 = Mode 1，说明 Video 中频在简单任务上贡献不大，可以在复杂任务上重新评估

**产出物**：RL 样本效率的量化对比，决定了三频架构的"第三条腿"是否成立。

---

## 每步与论文贡献的对应关系

```
实验 A (H1) → "可行性证明" — 独立论文可发，但不是核心贡献
实验 B (H2) → "核心创新" — AR 语义条件 > T5，三频 vs 双频的优势
实验 C (H3) → "工程贡献" — 实时推理的数字
实验 D (H4) → "方法贡献" — AR 作为自然语言闭环监督的新范式
实验 E (H5) → "最终贡献" — 一个自我提升的飞轮系统

理想的 Paper 架构:
  1. Introduction: 三频架构的动机和设计
  2. H2 实验: AR 条件信号的优势
  3. H3 实验: 多频推理的延迟
  4. H4 实验: AR 闭环监督的精度
  5. H5 实验: RL 的样本效率
  6. Ablation: Nano vs Super AR, with/without Video 中频, etc.
```

---

## 简版总结

| 阶段 | 实验 | 要回答的问题 | 成功标准 | 时间 |
|------|------|-------------|---------|:---:|
| P0 | AR NPU 推理 | AR 能独立工作吗？ | forward 不报错，输出合理 | 1-2d |
| P1 | AR vs T5 | 三频比 FastWAM 好在哪里？ | AR 条件信号 > T5 | 1-2w |
| P2 | 延迟基准 | 能实时部署吗？ | < 200ms action chunk | 1-2w |
| P3 | AR 偏差检测 | AR 能当闭环监督者吗？ | Precision>85%, Recall>90% | 1-2w |
| P4 | RL 效率 | 系统能自我提升吗？ | RL 样本效率 > BC | 3-6w |
