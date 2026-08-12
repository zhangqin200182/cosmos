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

### Step 0 A：一拆为二 + 复原验证（纯推理，零训练）

**目标**：将 Cosmos 3 拆为 AR + Generator 两部分，用 MoT mixed-attention 连回去，证明输出与拆分前一致。

**要新建的模块：0 个。所有参数来自 Cosmos 3 checkpoint。**

```
原始 Cosmos 3 (Policy 模式):
  image + text → [AR ↔ Generator, 共享 self-attn + add_*] → action + video

一拆为二:
  ① 提取 AR 塔 → prefill → AR K/V cache (36层)
  ② 提取 Generator 塔 → 读 AR K/V cache via MoT mixed-attention
  ③ 联合 forward → action + video

  对比: 拆分前 output == 拆分后 output ? (误差 < 1e-5)
```

**要解决的问题**：
1. 从 checkpoint 识别并分离 AR 参数和 Generator 参数
2. 实现 AR prefill + 36 层 K/V 缓存
3. 实现 MoT mixed-attention：Generator Q attend [AR_cached_K | Generator_K]
4. 验证数值等价性

**为什么应该等价**：Cosmos 3 中 AR tokens 被因果 mask 隔离，本来就看不到 DM tokens。AR 的 K/V 在原始和拆分两种情况下数学上应该完全一致。交叉注意力 `add_*` 是 DM→AR 单向，不改变 AR 的 K/V。**如果验证通过，就证明了拆分的正确性。**

**如果失败**：说明我们对架构的理解有偏差（存在未分析到的 AR←DM 交互路径）。需要深入排查。

---

### Step 0 B：一拆为三（Generator → Video + Action）

**目标**：在已验证正确的拆分架构上，将 Generator 进一步拆分为 Video DiT + Action DiT。

```
一拆为二（已验证 ✅）:
  AR 塔 + Generator 塔

一拆为三:
  AR 塔 + Video 中频（已有）+ Action 高频（新建）
  
  Action DiT:
    架构: 参考 FastWAM, hidden=1024, ffn=4096, 36 layers
    参数: ~300M（新建，需训练）
    训练: 冻结 AR + Video，仅训练 Action DiT + MoT 连接
```

**为什么分两步**：
- 0A 是纯验证——不引入新模块，确认拆分逻辑本身正确
- 0B 是建构——在已验证的基座上替换 Generator 的 action 部分
- 如果 0A 失败，0B 不可能对

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
Step 0A: 一拆为二（AR + Generator）
  → 0 个新模块, 纯推理, 零训练
  → 验证: 拆分后输出 == 原始 Cosmos 3 输出
  → 成本: 1-2 天

Step 0B: 一拆为三（AR + Video + Action）
  → 1 个新模块（Action DiT, ~300M）
  → 训练 Action DiT + MoT 连接
  → 成本: 1-2 周

Step 1: 实验验证
  → 1.1: AR vs T5 条件信号对比
  → 1.2: 延迟基准 (vs 原始 Cosmos 3, vs FastWAM)

Step 2: + Video 中频 → ablation 增益

Step 3: + AR 闭环 + RL → 样本效率

每一步的输出是下一步的输入。
每一步都有明确的成功标准和可对比的基线。
```
