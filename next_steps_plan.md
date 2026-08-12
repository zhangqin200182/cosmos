# 三频解耦架构 — 下一步验证计划

## 总体原则

按"风险最高、验证成本最低"的顺序推进。先证伪最致命的假设——如果某个核心假设在 Phase 1 就挂了，后面都不用做了。

## 核心假设清单（按可证伪性排序）

| # | 假设 | 如果这个错了意味着什么 | 验证成本 |
|---|------|----------------------|---------|
| H1 | AR 塔从 Cosmos 3 拆出后，VLM 能力不显著退化 | 整个方案的基础坍塌——AR 不能独立工作 | 低（1 天） |
| H2 | AR 的语义理解能产生比 T5 embedding 更好的 action 条件信号 | 三频相比 FastWAM 的核心优势不存在 | 低-中（2-3 天） |
| H3 | AR K/V + Video K/V prefill cache + Action-only 推理能显著降低延迟 | 多频推理的时延收益不成立 | 中（需要实现 MoT） |
| H4 | AR 闭环观察能正确判断执行偏差并给出有效修正 | Test-Time Scaling 和 RL 的核心前提不成立 | 中-高 |
| H5 | RL 训练的样本效率显著优于传统 RL baseline | "第三条腿"的独特价值不存在 | 高 |

---

## Phase 0：最小可行验证（1-2 周）

### 实验 0.1：AR 塔独立推理能力验证

**目标**：验证 H1——AR 从 Cosmos 3 拆分后是否保持 VLM 能力。

**方法**：

```python
# 1. 加载 Cosmos3-Nano 完整 checkpoint
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="auto"
)

# 2. 提取 AR 塔权重
# 保留: self_attn Q/K/V/O, mlp, embed_tokens, lm_head, vision_encoder, layernorms
# 丢弃: mlp_moe_gen, add_*, proj_in/out, time_embedder, action/audio 头

# 3. 在标准 VLM benchmark 上评估
# - Video Caption (MSRVTT / VATEX)
# - Temporal Localization
# - Embodied Reasoning (robot task planning)
# - Physical Plausibility Analysis

# 4. 对比 baselines
# - 原始 Cosmos3-Nano (完整模型，Reasoner 模式)
# - 原始 Qwen3-VL-8B (Cosmos 3 的起点)
```

**预期**：AR 能力 = 原始 Cosmos 3 Reasoner ≥ Qwen3-VL-8B。如果明显低于 Qwen3-VL-8B，说明 Cosmos 3 的 DiT 联合训练"污染"了 VLM 能力——这可能是一个重要的独立发现，但对三频架构不致命。

**致命信号**：AR 能力远低于原始 Cosmos 3 Reasoner（说明 `add_*` 等组件在 Reasoner 推理时承担了不可替代的功能，AR 不能独立工作）。

### 实验 0.2：AR vs T5 条件信号质量对比

**目标**：验证 H2——AR 的语义条件是否优于 T5。

**方法**：

```python
# 利用 FastWAM 的 Action DiT 作为统一测试平台
# 只换 text encoder:

# Variant A (FastWAM 原始): T5 text encoder → Action DiT
# Variant B (替换条件): Cosmos 3 AR 塔 → 冻结 → text embedding → Action DiT

# 在 LIBERO-10 / DROID 子集上对比:
# - Action prediction MSE (对 ground truth)
# - 任务成功率 (simulation)
# - 泛化能力 (unseen instructions)
```

**关键**：这是一个"零训练"实验——AR 塔不需要在机器人数据上微调，直接用它编码文本指令。如果 Variant B 在零训练条件下就能接近或超过 Variant A（FastWAM 的 T5），那就是非常强的信号。

**致命信号**：Variant B 明显差于 Variant A，且似乎不是因为"没训练"而是因为"embedding 不适合 diffusion 条件"（例如，即使加了简单的 MLP adapter 也对不齐）。

### 实验 0.3：FastWAM 基准复现 + Architecture 理解

**目标**：建立多频推理的工程基线，深入理解 MoT mixed-attention 的实现细节。

**方法**：

```bash
# 1. 复现 FastWAM 训练和推理 (LIBERO)
git clone FastWAM repo
python scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# 2. 测量 infer_joint vs infer_action 的延迟差异
# → 量化 prefill cache 的实际收益

# 3. 分析 MoT 代码 (mot.py) 的每个细节
# → 为 Phase 1 的 Cosmos 3 MoT 实现做准备
```

---

## Phase 1：核心工程验证（2-4 周）

### 实验 1.1：Cosmos 3 AR K/V Prefill + 虚拟 Action

**目标**：验证 prefill cache 机制在 Cosmos 3 权重上是否可行——不需要完整训练 Action DiT，只需要确认 AR 侧的 K/V 可以被缓存并正确读取。

**方法**：

```python
# 1. 加载 Cosmos3-Nano
# 2. AR Prefill: 编码文本 + 图像 → 跑 36 层 → 缓存每层的 K/V
# 3. 构造随机 Action tokens (模拟 Action DiT 的输出)
# 4. 尝试用 MoT mixed-attention: Action Q attend [AR_cached_K | Action_K]
# 5. 验证:
#    - 输出的 shape 是否正确
#    - 梯度是否可以通过 MoT mixed-attention 反向传播到 Action tokens
#    - AR K/V cache 是否在多次去噪步中保持稳定
```

**致命信号**：AR K/V 的形状或精度使得 MoT mixed-attention 无法正确连接（例如 head_dim 不匹配、RoPE 频率空间不兼容等）。

### 实验 1.2：最小 Action DiT + 机器人数据微调

**目标**：训练第一个可工作的三频原型，在 LIBERO 或 DROID 子集上验证端到端可行性。

**方法**：

```python
# 1. 构建 Action DiT (2-4 层即可，不需要完整 36 层)
# 2. AR Prefill: Cosmos 3 AR 塔 → 缓存 K/V
# 3. Video Prefill (可选): Cosmos 3 Generator 塔 → 缓存 K/V
# 4. MoT mixed-attention 连接
# 5. 在 ~1000 条机器人数据上微调
# 6. 测量: action prediction error, 推理延迟
```

**关键指标**：
- Action MSE 是否随训练下降（验证梯度可以正常流动）
- 推理延迟是否比完整的 Cosmos 3 Policy 推理有数量级提升
- Action 质量是否可以接受（哪怕不 SOTA，只要有学习信号）

### 实验 1.3：延迟基准测试

**目标**：用可测量的数字验证多频推理的时延收益。

| 配置 | 模型 | 参数 | 预期延迟 |
|------|------|------|---------|
| Baseline A | Cosmos 3 完整 Policy 推理 (Nano) | 16B | ~10-30s (50 steps) |
| Baseline B | Cosmos 3 完整 Policy 推理 (Nano, 4-step distilled) | 16B | ~1-3s |
| 三频 (仅 Action) | AR prefill + Action DiT | ~300M | 50-100ms/step |

---

## Phase 2：闭环验证（3-6 周）

### 实验 2.1：AR 偏差检测能力基准

**目标**：量化 AR 作为闭环监督者的判断精度——这是 Test-Time Scaling 和 RL 的前提。

**方法**：

```python
# 1. 构造一批"已知偏差"的轨迹:
#    - 正常轨迹 (ground truth)
#    - 注入已知误差的轨迹 (gripper 偏移、物体滑落、轨迹偏离等)
# 2. 让 AR 对每条轨迹做 supervised 判断（给 AR 看视频 + 任务描述）
# 3. 测量:
#    - 偏差检测精度 (precision/recall)
#    - 错误分类精度 (what went wrong?)
#    - 修正建议的合理性 (人工评估或 simulation 验证)
```

**致命信号**：AR 的偏差检测精度太低（< 70%）或假阳性太高（频繁错误触发重规划）。

### 实验 2.2：最小闭环执行

**目标**：跑通完整的"AR 低频→Video 中频→Action 高频"闭环，哪怕只在 1-2 个任务上。

**方法**：

```python
# 最小可行闭环:
# - AR: Cosmos 3 AR 塔 (prefill cache)
# - Video: 跳过 (先不做中频，降低复杂度)
# - Action: Phase 1 训练的 Action DiT
# - 调度: 简单的 fixed-frequency (每 50 steps AR 看一次)

# 在 LIBERO simulation 上跑端到端
# 测量: 成功率, AR 介入频率, 每次推理的延迟
```

---

## Phase 3：RL 验证（6-12 周）

### 实验 3.1：结果级 RL baseline

**目标**：验证 Mode 1 RL（结果级反馈）的有效性。

**方法**：
- Baseline: Phase 2 的 Action DiT 不做 RL
- RL: 用 AR 的 episodic critique 做 hindsight relabeling
- 比较: 在相同数据量下的成功率提升

### 实验 3.2：步骤级 RL + Imagination

**目标**：验证 Mode 2 RL（步骤级 + Video World Model）的增益。

**方法**：
- 加入 Video 中频的 imagination-based candidate scoring
- 比较 Mode 1 vs Mode 2 的样本效率

---

## 可以现在就开始做的低成本准备

### 1. 数据准备

```bash
# 下载 LIBERO dataset (FastWAM 格式)
huggingface-cli download yuanty/LIBERO-fastwam --local-dir ./data/libero

# 下载 Cosmos3-DROID (全套机器人数据)
huggingface-cli download nvidia/Cosmos3-DROID --local-dir ./data/droid
```

### 2. 代码基线搭建

```bash
# 1. Fork FastWAM → 跑通训练 → 理解每个模块
# 2. 跑通 Cosmos 3 Diffusers/Transformers 推理 → 理解 forward 流程
# 3. 提取 AR 塔权重 → 做 Experiment 0.1
```

### 3. Benchmark Suite 准备

```bash
# VLM 能力: VANTAGE-Bench, Video-MME, 或直接用 Cosmos 3 cookbook 的 reasoner examples
# Action 能力: LIBERO simulation, 或 DROID open-loop evaluation
# 延迟: 统一的 latency benchmark script (warmup + average over N runs)
```

---

## 决策树

```
Phase 0: AR 独立工作? 
  ├── YES → Phase 1: 训练 Action DiT + MoT
  │         ├── 延迟下降 10x+? 
  │         │   ├── YES → Phase 2: 闭环验证
  │         │   │         ├── AR 判断准? → Phase 3: RL → Paper!
  │         │   │         └── AR 判断不准 → 改进 prompt / 加轻量 classifier
  │         │   └── NO → 检查 MoT 实现 / 尝试更激进的蒸馏
  │         └── Action 不收敛 → 检查梯度流 / 简化架构
  └── NO → AR 能力退化严重
           ├── 原因: DiT 训练污染了 VLM? → 有趣发现, 可以写 paper
           └── 原因: 拆分丢失了关键组件? → 重新分析架构边界
```

---

## 最关键的三个问题（按优先级）

在开始任何代码工作之前，这三个概念性问题需要想清楚：

1. **MoT mixed-attention 的具体接口**：AR K/V cache 的形状是 [B, S_ar, head_dim×num_heads] per layer × 36 layers。Action DiT 的 hidden_dim=1024。这两个维度不匹配时——是加投影层对齐、还是让 Action DiT 也用 hidden_dim=4096（参数会增加到 ~2B）、还是只读 AR 的 K/V 而不读 V？这个选择决定了整个 MoT 层的参数量和复杂度。

2. **Video 中频是否从 Phase 1 就设计进去**：如果先做双频（AR + Action），Video 作为可选项后加，架构设计上应该预留"第三 expert 插槽"——MoT mixed-attention 的接口应该支持 `n_experts ≥ 2`，而不是硬编码为 2。

3. **AR 的结构化输出格式**：用 constrained decoding（guidance / outlines）保证 JSON 格式，还是接受 AR 自由文本输出的不稳定？前者增加工程复杂度，后者可能在实际部署中不可靠。可以先做自由文本，在 benchmark 上测量 JSON 解析成功率——如果 > 95%，暂时不需要 constrained decoding。
