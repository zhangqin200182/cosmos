# 三频架构：构建路径与实验验证

---

## 整体路线

```
Phase 1: Cosmos 3 复现 → 跑通原始模型推理
    ↓
Phase 2: 一拆为二 → 验证拆分后仍能 work
    ↓
Phase 3: 一拆为二的收益验证 → AR/DiT 不同频率 + AR 观测反馈
    ↓          （不引入任何新模块，纯用已有权重）
    ↓
Phase 4: 一拆为三 → 训练 Action DiT → 完整收益验证
```

---

## Phase 1：Cosmos 3 复现

**目标**：在 NPU 上跑通 Cosmos 3 原始推理，建立基线。

**要做的**：
- 加载 Cosmos3-Nano checkpoint
- 跑通 Reasoner 模式（text/image → text）
- 跑通 Generator Policy 模式（image + text → action + video）
- 记录基线指标：推理延迟、显存占用、输出质量

**产出**：NPU 上的 Cosmos 3 推理基线。后续所有"拆分后 vs 拆分前"的对比，都以这个基线的输出为准。

---

## Phase 2：一拆为二（AR + Generator）

**目标**：将 Cosmos 3 拆成 AR 塔和 Generator 塔，用 MoT mixed-attention 连回去，验证输出与拆分前一致。

**一拆为二是什么**：

```
原始 Cosmos 3:
  [AR 塔] ←→ [Generator 塔]
  共享 self-attn Q/K/V + add_* 交叉注意力

拆分后:
  ┌──────────┐         ┌───────────────────┐
  │  AR 塔    │         │  Generator 塔      │
  │ (Reasoner)│         │ (Video + Action)  │
  │           │ AR K/V  │                   │
  │ prefill → │────────→│ 读 AR K/V cache   │
  │ K/V cache │         │ (MoT连接)          │
  └──────────┘         └───────────────────┘

  0 个新模块。
  所有参数来自 Cosmos 3 checkpoint。
  只需实现 AR prefill + K/V cache + MoT mixed-attention 连接。
```

**要解决的问题**：
1. 从 checkpoint 识别并分离 AR 参数和 Generator 参数
2. 实现 AR prefill → 36 层 K/V 缓存
3. 实现 MoT mixed-attention：Generator Q attend [AR_cached_K | Generator_K]
4. 验证数值等价性：拆分后输出 == 原始 Cosmos 3 输出（误差 < 1e-5）

**为什么应该等价**：AR tokens 被因果 mask 隔离，本来就看不到 DM tokens。AR 的 K/V 在拆分前后数学上完全一致。

---

## Phase 3：一拆为二的收益验证

**关键洞察**：一拆为二后，AR 和 Generator 已经是两个独立运行的模块，收益自然显现——不需要等到一拆为三。

### 收益 3.1：多频推理（不同执行频率）

```
原始 Cosmos 3:
  每步去噪都跑 AR FFN + Generator FFN → AR FFN 50 次重复计算

一拆为二后:
  AR prefill 一次 (缓存 K/V) → Generator 去噪 N 步，每次只跑 Generator FFN
  → AR 的计算开销从 N 次降到 1 次
  → Generator 的每步去噪更快（不再夹杂 AR FFN）
```

**实验**：对比拆分前后的 Policy 推理延迟。

| 配置 | 每步计算 | N=50 总延迟 | N=4 总延迟 |
|------|---------|:---:|:---:|
| 原始 Cosmos 3 | AR FFN + Gen FFN | T_original × 50 | T_original × 4 |
| 一拆为二 | Gen FFN only (AR cached) | T_split × 50 | T_split × 4 |

**预期**：T_split < T_original（Gen FFN 不再被 AR FFN 的计算和同步拖累）。即使不去蒸馏，拆分本身就有加速。

### 收益 3.2：AR 观测与反馈（低频 AR 监督中频 DiT）

```
AR 不需要每步都运行。它可以"偶尔看一眼"Generator 的输出：

  循环中:
    Generator 生成 action + video rollout（中频, ~3 Hz）
    AR 每 N 步醒来一次（低频, ~1 Hz）:
      → 看 Generator 生成的 video rollout
      → 判断: 轨迹是否合理？目标是否在接近？
      → 如果不合理 → 更新文本条件 → 指导 Generator 调整
      → 如果子目标完成 → 推进到下一阶段
```

**实验**：在一个闭环任务上（如 LIBERO simulation），对比：

| 条件 | AR 频率 | 描述 |
|------|:---:|------|
| Baseline | 每步 | 原始 Cosmos 3 Policy（AR 每步都跑） |
| 多频-稀疏 | 每 10 步 | AR 每 10 步看一眼，其余步只用缓存条件 |
| 多频-自适应 | 动态 | AR 自己决定什么时候看（基于上次评估的置信度） |

**对比指标**：
- 任务成功率（多频不应该明显低于每步 AR）
- AR 总计算量（多频应远低于每步 AR）
- AR 成功发现并纠正偏差的次数

**核心问题**：AR 低频运行会不会导致任务成功率下降？在什么频率下能兼顾效率和效果？

---

## Phase 4：一拆为三 + 收益验证

**目标**：将 Generator 进一步拆为 Video 中频 + Action 高频，训练轻量 Action DiT。

```
一拆为二:
  AR 塔 + Generator 塔 (Video + Action 在一起)

一拆为三:
  AR 塔 + Video 中频 (已有) + Action 高频 (新建, ~300M, 需训练)
```

### 构建

- 从 Generator 塔中分离 Video 能力（保留）
- 构建轻量 Action DiT（参考 FastWAM, hidden=1024, ffn=4096）
- 训练：冻结 AR + Video，只训练 Action DiT + MoT 连接

### 收益验证

| 实验 | 对比 | 要证明什么 |
|------|------|-----------|
| 4.1: AR vs T5 | 三频架构 vs FastWAM（相同 ActionDiT，不同条件编码器） | AR 的条件信号优于 T5 |
| 4.2: 延迟基准 | 三频 Action-only vs 一拆为二的 Generator | Action 高频推理的延迟优势 |
| 4.3: Video 增益 | 有无 Video 中频的 ablation | Video 物理验证的价值 |
| 4.4: RL 样本效率 | RL vs BC 微调 | AR 监督信号使系统自我提升 |

---

## 各阶段的模块变化

```
Phase 1:  原始 Cosmos 3 推理
          无拆分，无新模块

Phase 2:  一拆为二
          AR 塔 (已有) + Generator 塔 (已有)
          新建: MoT 连接 (~200 行 Python)
          新训练: 0

Phase 3:  一拆为二 + 多频调度
          同 Phase 2 的模块，加上:
          新建: AR 周期性唤醒逻辑 (~100 行 Python)
          新训练: 0

Phase 4:  一拆为三
          AR 塔 (已有) + Video 中频 (已有) + Action 高频 (新建, ~300M)
          新建: Action DiT 网络 + 训练管线
          新训练: Action DiT (LIBERO/DROID, 数万条数据)
```

---

## 收益的逐步呈现

```
Phase 2 完成时:
  ✓ 拆分本身被验证为 lossless

Phase 3 完成时（一拆为二 + 多频调度）:
  ✓ 收益 A: 推理加速（AR 从 N 次降到 1 次）
  ✓ 收益 B: AR 低频观测 + 反馈（test-time scaling 的雏形）
  → 此时已有可发表的阶段性结果

Phase 4 完成时（一拆为三）:
  ✓ 收益 C: Action 高频 → 实时控制延迟
  ✓ 收益 D: AR 条件信号 > T5
  ✓ 收益 E: Video 中频物理验证
  ✓ 收益 F: RL 自我提升
```
