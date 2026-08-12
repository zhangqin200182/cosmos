# 三频架构：构建路径与实验验证（修订版）

> **核心修正**：加入 Step 0——先证明拆分本身是 lossless 的，再在拆分后的架构上验证收益。

---

## 一、最终架构

```
模块 A: AR 低频推理器         ← Cosmos 3 checkpoint 拆分（已有）
模块 B: Video 中频想象器      ← Cosmos 3 checkpoint 拆分（已有，可选）
模块 C: Action 高频执行器     ← 新建（需训练）

连接机制: MoT Mixed-Attention ← 新建（需实现）

最终系统 = A + B + C + MoT 连接
```

---

## 二、构建与验证路径

### Step 0：拆分 + 复原验证（Fidelity Check）

**目标**：证明将 Cosmos 3 拆成三部分再连起来后，输出与拆分前完全一致。

```
原始 Cosmos 3 (Policy 模式):
  image + text → [统一 forward] → action + video

拆分后:
  ① 提取 AR 塔 → prefill → AR K/V cache
  ② 提取 Generator 塔 → 读取 AR K/V cache
  ③ 实现 MoT mixed-attention 连接
  ④ 联合 forward → action + video

对比: 拆分前 output vs 拆分后 output
  → 数值误差 < 1e-5 (bf16 精度范围内的浮点等价)
  → 生成的动作序列和视频帧完全一致
```

**要解决的问题**：
1. 从 checkpoint 中识别并分离 AR 塔参数（哪些属于 AR，哪些属于 Generator，哪些共享）
2. 实现 AR prefill + K/V 缓存
3. 实现 MoT mixed-attention（替代原版 self-attention 中的 DM→AR 通路）
4. 验证数值等价性

**关键问题**：MoT mixed-attention 的连接方式会和原始 Cosmos 3 的 self-attention 有细微差异（原始是 Q/K/V 全部在同一次 softmax 中，拆分后是 Action Q attend cached AR K/V）。这两种计算路径是否数学等价？如果不完全等价（例如原始 attention 中有 DM→AR 的交互会影响 AR 的输出，但拆分后 AR 是 prefill 的，不受 DM 影响），差异有多大？

**实际上 Cosmos 3 的 AR tokens 本身就看不到 DM tokens**（因果 mask 阻止 + 交叉注意力单向）。所以 AR 的 K/V 在原始和拆分两种情况下**应该完全一致**——AR 的 forward 不依赖 DM tokens。如果验证通过，这就证明了拆分是 lossless 的。

**成功标准**：拆分前 vs 拆分后的 action 输出误差 < 1e-5（浮点等价）。

**如果失败**：说明 AR 和 DM 之间的交互比架构分析显示的更紧密（可能 DM tokens 通过某些未分析到的路径影响了 AR）。需要重新分析信息流。

---

### Step 1：双频训练（AR + Action）

**目标**：在已验证 lossless 的拆分架构上，训练 Action DiT，实现双频推理。

```
基于 Step 0 验证的拆分方式:
  ① AR 塔: 冻结 → prefill K/V cache
  ② Action DiT: 新建 + 训练 ← 读取 AR K/V cache
  ③ MoT mixed-attention: 同 Step 0 的实现

训练: 冻结 AR + MoT 连接权重，只训练 Action DiT
```

**为什么 Step 0 必须先做**：
- Step 0 验证了 AR K/V cache + MoT 连接的正确性
- Step 1 只是把 Generator 塔替换为 Action DiT，AR 侧完全不动
- 如果 Step 0 验证通过，Step 1 的 AR 侧推理正确性就有了保证

---

### 实验 1.1：AR vs T5（双频 vs FastWAM）

**控制变量**：相同的 Action DiT 架构和训练数据，唯一不同是条件编码器。

| 条件 | Variant A (FastWAM) | Variant B (我们的双频) |
|------|:---:|:---:|
| 文本编码器 | T5 | Cosmos 3 AR 塔 |
| Action DiT | FastWAM ActionDiT | 我们的 ActionDiT |
| 训练数据 | LIBERO-10 | 同 A |
| 训练量 | 同 A | 同 A |

**对比指标**：Action MSE、任务成功率、未见指令泛化率。

**核心问题**：AR 的语义理解和物理推理能力，是否转化为更好的 Action 条件信号？

---

### 实验 1.2：延迟对比

| 配置 | 预期延迟 / Action Chunk | 等效频率 |
|------|:---:|:---:|
| Cosmos 3 原始 Policy (16B, 50步) | ~10-30s | < 1 Hz |
| **我们的双频 (AR prefill + Action, N步)** | **< 200ms** | **> 30 Hz** |

---

### Step 2：加入 Video 中频（可选）

**目标**：在双频基础上加入 Video 层，做 physical plausibility gate。

**实验 2.1**：有无 Video 中频的 ablation。在需要物理推理的复杂任务上对比双频和三频的成功率和 AR 触发频率。

---

### Step 3：AR 闭环 + RL

**目标**：AR 定期观察执行效果 → 判断偏差 → 积累训练数据 → RL 自我提升。

**实验 3.1**：AR 偏差检测精度。**实验 3.2**：RL vs BC 样本效率。

---

## 三、Paper 叙事线

```
Step 0: 拆分证明 (Fidelity)
  → "Cosmos 3 的 AR 和 DM 在推理时可以被无损分离"
  → 建立三频架构的正确性基础

实验 1.1: AR 条件信号 > T5
  → "AR 的语义理解优势转化为更好的 Action 生成"

实验 1.2: 延迟基准
  → "多频推理实现实时控制"

实验 2.1: Video 增益
  → "中频物理验证在复杂任务上的价值"

实验 3.1+3.2: 闭环 + RL
  → "AR 自然语言监督使系统自我提升"
```

---

## 总结：构建顺序

```
Step 0: 拆 Cosmos 3 → 连回去 → 证明输出不变（fidelity check）
   ↓
Step 1: 训练 Action DiT → 双频系统 → 验证 AR > T5 + 验证实时延迟
   ↓
Step 2: 加 Video 中频 → 三频系统 → 验证 Video 的 ablation 增益
   ↓
Step 3: 加 AR 闭环 + RL → 自我提升系统 → 验证 RL 样本效率

每一步的输出是下一步的输入。
每一步都有明确的成功标准和可对比的基线。
```
