# Cosmos 3 三频解耦架构：多频推理、Test-Time Scaling 与强化学习的统一框架

> 基于对 NVIDIA Cosmos 3 MoT（Mixture-of-Transformers）架构的深入分析，结合 FastWAM 的多频推理思路，提出将统一模型按"认知频率"拆分并协同运作的研究方向。

---

## 目录

1. [背景：Cosmos 3 MoT 架构深度分析](#1-背景cosmos-3-mot-架构深度分析)
2. [FastWAM 的启示：多频推理的可行性](#2-fastwam-的启示多频推理的可行性)
3. [三频解耦架构设计](#3-三频解耦架构设计)
4. [闭环监督：AR VLM 作为系统观察者](#4-闭环监督ar-vlm-作为系统观察者)
5. [收益一：多频推理解决时延问题](#5-收益一多频推理解决时延问题)
6. [收益二：Test-Time Scaling 动态纠错](#6-收益二test-time-scaling-动态纠错)
7. [收益三：强化学习使系统自我提升](#7-收益三强化学习使系统自我提升)
8. [三条收益的协同增强](#8-三条收益的协同增强)
9. [实现路线图](#9-实现路线图)
10. [待解决的关键问题](#10-待解决的关键问题)

---

## 1. 背景：Cosmos 3 MoT 架构深度分析

### 1.1 架构总览

Cosmos 3 是一个全模态世界模型，基于统一的 Mixture-of-Transformers (MoT) 架构，能够同时处理和理解语言、图像、视频、音频和动作五种模态。模型有三个规模：Super (64B)、Nano (16B) 和 Edge (4B/2B dense)。

核心创新：同一个 Transformer 骨干支持两种运行模式——Reasoner（自回归因果注意力，做理解和推理）和 Generator（全注意力扩散去噪，做生成）。

### 1.2 权重层面的真实架构

通过分析 `nvidia/Cosmos3-Nano` 的实际 safetensors 权重结构，每层（共36层）包含：

```
layers.{N}.
├── self_attn.
│   ├── to_q.weight          ← Q 投影 ┐
│   ├── to_k.weight          ← K 投影 │ 只有一套！AR 和 DM 共享
│   ├── to_v.weight          ← V 投影 │
│   ├── to_out.weight        ← 输出   ┘
│   │
│   ├── add_q_proj.weight    ← 交叉 Q  ┐
│   ├── add_k_proj.weight    ← 交叉 K  │ 交叉注意力（DM→AR 单向）
│   ├── add_v_proj.weight    ← 交叉 V  │ 专用于条件注入
│   ├── to_add_out.weight    ← 输出   ┘
│   │
│   ├── norm_q.weight / norm_k.weight         ← QK Norm
│   └── norm_added_q.weight / norm_added_k.weight
│
├── mlp.                     ← AR FFN（理解用）     ┐
│   ├── gate_proj.weight                            │
│   ├── up_proj.weight                              │ 两套独立 FFN
│   └── down_proj.weight                            │
│                                                   │
├── mlp_moe_gen.             ← Generator FFN（生成用）│
│   ├── gate_proj.weight                            │
│   ├── up_proj.weight                              ┘
│   └── down_proj.weight
│
├── input_layernorm / post_attention_layernorm          ← AR 侧
└── input_layernorm_moe_gen / post_attention_layernorm_moe_gen ← Generator 侧
```

关键发现：
- **Self-Attention Q/K/V/O 只有一套**，AR 和 DM 共享。两种模式的区别仅在于注意力 mask（因果 vs 全注意力）。
- **FFN 是两套独立的**（`mlp` 处理 AR tokens, `mlp_moe_gen` 处理 DM tokens），按 token 类型路由。
- **交叉注意力 `add_*` 是额外的**，尺寸和 self-attention 完全对称（每层 42M 参数），用于 DM→AR 单向条件注入。
- **信息流是非对称的**：DM tokens 可以通过 self-attention（全注意力 mask）和 cross-attention（add_*）两条通路读取 AR 信息；AR tokens 被因果 mask 隔离，看不到 DM tokens。

### 1.3 每层参数量分布（Cosmos3-Nano, hidden=4096, intermediate=12288）

| 组件 | 参数量/层 | 占比 |
|------|-----------|------|
| Self-Attention Q/K/V/O + Norms | ~42M | 10.9% |
| Cross-Attention add_q/k/v/o + Norms | ~42M | 10.9% |
| AR FFN (gate/up/down) | ~151M | 39.1% |
| Generator FFN (gate/up/down) | ~151M | 39.1% |

**FFN 占总参数量的 78.2%**，attention 占 21.8%。这意味推理的主要计算瓶颈在 FFN。

### 1.4 训练目标与方法

Cosmos 3 围绕三大训练目标设计，形成了三阶段训练管线：

| 训练目标 | 模式 | 目标函数 | 数据需求 |
|----------|------|----------|----------|
| 视觉生成 | Generator (DiT) | Flow Matching / Diffusion Loss | ~400M 视频 + ~1B 图像 |
| 动作建模 | Generator (DiT) | Flow Matching（Policy/Forward/Inverse Dynamics） | DROID, LIBERO, AV 等机器人数据 |
| 世界推理 | Reasoner (AR) | Next-Token Prediction（交叉熵） | 视觉问答, 物理推理数据 |

**训练流程**：从 Qwen3-VL-8B VLM 权重初始化 → 复制 FFN 作为 Generator FFN 初始值（两者结构完全一致） → Generator 预训练（冻结 AR FFN，训练 DM FFN + 交叉注意力 + 投影头）→ 后训练 SFT（解冻全部，联合微调）。

关键设计：Generator 不是从现成 DiT 训练而来，而是从 VLM "长"出来的——VLM 的 self-attention Q/K/V 被直接复用，VLM 的 mlp 被复制为 mlp_moe_gen 的初始化。这意味着 Generator 的 attention 能力完全继承自 VLM 的视觉-语言理解能力。

### 1.5 训练中的 AR↔DiT 相互受益关系

**AR → DiT（直接且巨大）**：

- 语言理解：AR 训练教会的语义通过交叉注意力直接注入 DM token，使动作生成精确跟随文本条件
- 视觉理解：AR 学会的物体边界、材质、空间关系通过 self-attention 中的 DM→AR 通路被 DM 读取
- 物理常识：共享的 self-attention Q/K/V 编码了物理世界规律，使 DM 生成的动作和视频自动符合物理约束
- 时序因果：AR 因果注意力训练的能力使 DM 生成的动作序列有正确的顺序依赖

**DiT → AR（间接且微妙）**：
- AR 不能 attend DM tokens（因果 mask 阻止 + 交叉注意力单向），因此推理时无直接帮助
- 仅在训练过程中通过共享 Q/K/V 的梯度协同传递——视频动态表征和细粒度视觉特征通过共享投影间接提升 AR 的视觉理解能力
- 效果不对称是架构的刻意设计

### 1.6 关于消融证据的诚实评估

Paper 公开版（arXiv:2606.02800）**没有**提供教科书式的对照消融表。NVIDIA 通过以下方式论证了统一架构的有效性：
1. **基准横扫**：同一 checkpoint 在 VANTAGE-Bench、PAI-Bench、RoboLab、Physics-IQ 等理解和生成赛道同时登顶
2. **训练流程设计**：`freeze_und: false` 的默认配置、从 VLM 初始化的决策、交叉注意力 42M/层的参数预算——这些工程选择暗示内部做过消融
3. **DuoNeural 社区意外消融**：修改 AR 通路晚期层（15-32）的 201/814 个权重（24.7%）→ 生成了安全内容的质量下降，反向验证了 AR↔DM 耦合的真实性

缺失的消融：仅 DiT（无 AR 塔）、AR 随机初始化（非 VLM）、去掉交叉注意力（仅 self-attention）、分开训练（独立 VLM + 独立 DiT）等对照实验的数字未公开。

---

## 2. FastWAM 的启示：多频推理的可行性

### 2.1 FastWAM 简介

FastWAM（"Do World Action Models Need Test-time Future Imagination?"）是一个面向机器人操作的世界-动作模型。它的核心贡献是证明了"慢速视频生成 + 快速动作生成"的双频架构是可行的。

### 2.2 FastWAM 架构

```
Video Expert (Wan2.2 DiT backbone)     Action Expert (ActionDiT)
  hidden=3072, ffn=14336                  hidden=1024, ffn=4096
  30 layers, ~5B params                   30 layers, ~300M params
  ──────────────────────                  ────────────────────
  各自的 self-attn Q/K/V/O                各自的 self-attn Q/K/V/O
  各自的 cross-attn (对 text ctx)          各自的 cross-attn (对 text ctx)
  各自的 FFN                               各自的 FFN
  各自的时间调制                            各自的时间调制
         │                                       │
         └────── MoT Mixed-Attention ────────────┘
              每层: cat([Q_v, Q_a]) attend cat([K_v, K_a], [V_v, V_a])
              然后各自跑 post-block (FFN + cross-attn)
```

### 2.3 FastWAM 的多频调度

FastWAM 提供两种推理模式：

**infer_joint（联合推理）**：Video + Action 一起去噪，生成完整视频 rollout + 动作序列。用于训练和评估。

**infer_action（仅动作推理）**：
1. **Prefill 阶段**（慢，触发一次）：将当前帧通过 Video Expert 编码为 latent，跑完 30 层，缓存每层的 K/V
2. **Action 去噪阶段**（快，N 步循环）：Action Expert 通过 MoT mixed-attention 读取 Video K/V cache，只跑 Action Expert 的 FFN 和 cross-attention
3. Video Expert 不重算 → 每次去噪只跑 ~300M 参数 → 延迟从秒级降到百毫秒级

**这是 FastWAM 能实现实时机器人控制的核心机制。**

### 2.4 FastWAM MoT vs Cosmos 3 MoT：架构差异

| 维度 | Cosmos 3 | FastWAM |
|------|----------|---------|
| Self-Attention | 1 套共享 Q/K/V/O | N 套独立 Q/K/V/O（每 expert 一套） |
| 参数组织 | Token 级路由（AR token → mlp, DM token → mlp_moe_gen） | Expert 级路由（整个 block 都是独立的） |
| 交叉注意力 | `add_*` 投影，DM→AR 单向，专做条件注入 | 每 expert 内部的 cross-attn，对 text context |
| Attention Mask | AR: causal; DM: full + 额外 cross-attn | per-expert mask: first_frame_causal / group_diagonal |
| 推理调度 | 联合推理（所有 token 一起） | 可分离推理（prefill cache + action-only） |
| 设计动机 | 训练侧优化（统一表征学习） | 推理侧优化（多频调度） |

---

## 3. 三频解耦架构设计

### 3.1 核心理念

将 Cosmos 3 的统一 MoT 模型按"认知频率"拆分为三个独立运行的模块，利用共享的 self-attention Q/K/V/O 和多频 MoT mixed-attention 实现高效的信息交换：

```
推理频率:     ~0.5-1 Hz              ~2-5 Hz                ~20-50 Hz
延迟要求:     1-2秒                  200-500ms              20-50ms
═══════════════════════════════════════════════════════════════════════

          ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
          │   AR 低频     │    │  DiT 中频     │    │ Action 高频   │
          │   (Reasoner)  │    │  (Video Gen)  │    │ (Action Gen)  │
          ├──────────────┤    ├──────────────┤    ├──────────────┤
权重来源:  │ mlp (保留)    │    │ mlp_moe_gen   │    │ 新建 ActionDiT│
          │ self_attn QKV │    │ self_attn QKV │    │ (独立 Q/K/V)   │
          │ vision_enc    │    │ add_* cross   │    │ ~300M params   │
          │ lm_head       │    │ proj_in/out   │    │                │
          │               │    │ VAE           │    │                │
          │               │    │ time_embed    │    │                │
          ├──────────────┤    ├──────────────┤    ├──────────────┤
职责:     │ 理解+规划      │    │ 想象+验证      │    │ 执行           │
          │ "为什么做"     │    │ "这样做对吗"    │    │ "具体怎么做"    │
          ├──────────────┤    ├──────────────┤    ├──────────────┤
输入:     │ 当前图像       │    │ 当前图像       │    │ 当前图像        │
          │ 任务指令       │    │ AR 文本计划    │    │ AR 文本计划     │
          │ 历史上下文     │    │ 动作序列(可选)  │    │ Video latent    │
          ├──────────────┤    ├──────────────┤    │ (来自中频)      │
输出:     │ 子目标/计划    │    │ 未来 N 帧 rollout│   │ action chunk   │
          │ 场景理解       │    │ (latent 即可)  │    │ [a1...a32]     │
          │ 偏差分析       │    │                │    │ + uncertainty   │
          ├──────────────┤    ├──────────────┤    ├──────────────┤
KV Cache: │ prefill 一次   │    │ AR K/V (读)    │    │ AR K/V (读)    │
(信息流)   │ → 供中频+高频读 │    │ Video K/V (写) │    │ Video K/V (读)  │
          │               │    │ → 供高频读      │    │ Action K/V(写) │
          └──────────────┘    └──────────────┘    └──────────────┘
                  │                   │                   │
                  └───────────────────┴───────────────────┘
                          通过 MoT Mixed-Attention 交互
                    (保留 Cosmos 3 共享 self-attn 或用 FastWAM 式拼接)
```

### 3.2 三个频率各自的职责

**低频 AR（~1 Hz）—— "思考者"**

- 触发时机：任务开始、子目标切换、检测到显著偏差、超时重规划
- 输入：当前帧、任务指令、上次规划时的关键帧和预期描述
- 输出：结构化的子目标链、每次决策下次检查的时机（`next_check_after_steps`）、对 Action 执行的评估
- 核心价值：不仅产出"做什么"，还产出"什么时候再来看"——AR 自己决定自己的唤醒频率

**中频 Video DiT（~3 Hz）—— "想象者"**

- 触发时机：AR 产出新计划后（验证可行性）、预测误差积累到阈值、周期性自检
- 输入：当前帧 latent、AR 文本条件 embedding、候选动作序列
- 输出：未来 16-32 帧视频 rollout（latent space 即可，不一定解码到像素）
- 核心价值：
  1. 自验证：rollout 中机器人手臂是否在物理约束内？物体是否稳定？是否接近碰撞？
  2. 为高频 Action 提供额外的视觉条件信号（video latent 直接注入）
  3. 在 imagination 中评估多个候选动作序列

**高频 Action DiT（等效 ~30 Hz）—— "执行者"**

- 触发时机：连续运行，异步重规划
- 输入：当前帧（仅第一帧即可）、AR 文本条件、中频 Video latent（可选）
- 去噪步骤：4-8 步（蒸馏后极少量）
- 输出：action chunk [a1, ..., a32]，执行前 8 步后异步重推理
- 等效控制频率 = 32 / 推理延迟

### 3.3 同步协议：基于信息差而非时钟

三个频率不是按固定时间间隔触发，而是基于"信息差"——当某一层产生意外信息时向上抛，当某一层做出新决策时向下推：

```
时间线:
─────────────────────────────────────────────────────────────→

AR:    [────plan────]                    [──replan──]         [──replan──]
              ↓                                ↑                  ↑
              ↓                     (子目标完成检测 → 确认)    (周期性)
              ↓                                ↑                  ↑
Video:        [──rollout──]   [──rollout──]   [──rollout──]
                   ↓               ↓               ↓
                   ↓    video latents 作为条件     ↓
                   ↓               ↓               ↓
Action: [chunk][chunk][chunk][chunk][chunk][chunk][chunk][chunk][chunk]
         ← 32步→  ← 32步→                          ← 32步→
         执行8步   执行8步                              执行8步
```

五种自然触发机制：

1. **预测误差触发**（Bottom-up）：Action 预测的下一步状态 vs 实际观测的偏差 → 小偏差忽略，中偏差触发 Video 检查，大偏差直接触发 AR 重规划
2. **不确定性触发**：Action 去噪最终步的 latent 方差大 → 模型自己"知道"自己不确定 → 主动请求上层检查
3. **子目标完成触发**（Top-down + Bottom-up）：AR 在规划时定义了每个子目标的 "completion_condition" → Action 和 Video 持续检查 → Video 先验证 → AR 确认并推进
4. **物理合理性门控**：Video 中频每次 rollout 自动检测碰撞、稳定性、关节限制 → 全部通过则继续，失败则先重试（不同噪声种子），多次失败则触发 AR
5. **信息新鲜度**：跟踪 AR 上次规划时看到的图像是多久前的（`age_of_last_observation_used_in_plan`）→ 世界可能已经变了 → 主动重规划

### 3.4 从 Cosmos 3 权重出发的具体实现步骤

**Step 1**：提取 AR 通路作为固定条件编码器
- 保留 `self_attn Q/K/V/O`、`mlp`、`embed_tokens`、`vision_encoder`
- 保留 `add_*` 交叉注意力（供 Video 中频和 Action 高频读取条件）
- 推理时：prefill 一次，缓存所有 36 层的 K/V

**Step 2**：保留 Video 中频
- 保留 `mlp_moe_gen`、`add_*`、`proj_in/out`、`VAE`、`time_embedder`
- 保留 AR 的 `add_*` K/V 读取能力

**Step 3**：新建 Action DiT
- 结构参考 FastWAM ActionDiT：`hidden=1024, ffn=4096, 36 layers`（与 Cosmos 3 的 num_heads/attn_head_dim 对齐）
- 独立的自注意力 Q/K/V/O（随机初始化，或从 Cosmos 3 的 `mlp_moe_gen` 蒸馏初始化）
- 通过 MoT mixed-attention 读取 AR K/V cache 和 Video K/V cache
- `~300M` 参数

**Step 4**：实现 MoT Mixed-Attention 层
- 参考 FastWAM `mot.py` 的实现模式
- 每层：concat [AR_cached_K | Video_cached_K | Action_K]，Action Q attend 全部
- Action 有自己的 post-attention FFN + cross-attention
- AR 和 Video 侧不需要 post-attention（已预填充）

**Step 5**：训练
- 阶段 1：冻结 AR + Video，只训练 Action DiT（action MSE loss，几万-几十万条机器人数据）
- 阶段 2（可选）：解冻 Video 部分参数（LoRA），联合训练 Action + Video（加轻量 video consistency loss）
- 阶段 3（RL）：AR 监督的闭环自我提升

---

## 4. 闭环监督：AR VLM 作为系统观察者

### 4.1 核心洞察

**不需要外挂 Gate 来协调三频调度。AR VLM 本身就是最聪明的观察者和决策者。** 它可以：
- 看图像（观测执行效果）
- 比较预期和实际（评估偏差）
- 判断是否需要调整、重规划或继续
- 决定自己下次什么时候醒来

### 4.2 AR 的结构化输出

```json
{
  "observation": "杯子在 gripper 中, 位置离 shelf 上方约 5cm。轨迹正常。",

  "deviation_assessment": {
    "level": "minor",
    "description": "杯子比预期偏右约 2cm, 可能是前几步的累积漂移"
  },

  "subgoal_progress": {
    "current_subgoal_id": 3,
    "is_complete": false,
    "progress_pct": 85,
    "evidence": "杯子已到达 shelf 上方 85% 位置, 但尚未完全对齐"
  },

  "safety_check": {
    "collision_risk": false,
    "object_stability": "stable",
    "any_anomaly": false
  },

  "decision": {
    "action": "adjust",
    "reasoning": "无需完整重规划, 只需告诉 Action 向左修正 2cm",
    "if_adjust": "向左侧修正平移方向约 2cm, 保持其他参数不变"
  },

  "next_check": {
    "after_steps": 30,
    "or_when": "杯子到达 shelf 正上方 1cm 以内",
    "reasoning": "预计约 30 步到达, 当前状态稳定, 无需频繁检查"
  }
}
```

`next_check.after_steps` 字段是**唯一的调度策略**——AR 自己决定自己下次什么时候醒来。

### 4.3 AR 的自然分层响应

AR 看到不同情况时，自然地产生不同粒度的响应：

| 情况 | AR 判断 | 响应 | 对下层的影响 |
|------|---------|------|-------------|
| 一切正常 | `decision: continue` | 设定下一次检查时机 | Action 继续，Video 按需自检 |
| 小偏差 | `decision: adjust` | 轻量修正指令 | Action 调整方向，不重规划 |
| 子目标完成 | `decision: advance_subgoal` | 推进到下一子目标，给出新执行细节 | Action 切换任务模式 |
| 显著偏差 | `decision: replan` | 完整重规划 | Action 重新开始 |
| 不可恢复 | `decision: abort` | 终止任务 | 全部停止 |

### 4.4 和人工设计 Gate 的对比

| 维度 | 外挂 Gate（规则/分类器） | AR VLM 自我监督 |
|------|--------------------------|-----------------|
| 预测误差检测 | 手动标定阈值 | AR 看图像直接判断 |
| 物理合理性 | 训练分类器（~500K params） | AR 自然推理（已具备物理常识） |
| 子目标完成 | 训练 detector | AR 看图像 → 比较预期 |
| 不确定性估计 | Action 附带 uncertainty head | AR 综合判断 |
| 重规划时机 | 规则函数 | AR 在自己的 think 里决定 |
| 跨任务泛化 | 每个任务重新设计 | AR 的通用理解→零样本迁移 |

**结论**：唯一的"外挂"是超时唤醒（如果 AR 指定的 `next_check_after_steps` 到了但 Action 还在执行，强制唤醒 AR）。这是一个纯规则，不需要训练。

---

## 5. 收益一：多频推理解决时延问题

### 5.1 问题

Cosmos 3 原始架构中，AR FFN 和 Generator FFN 在**每次去噪步**都完整执行。AR 通路每步输出完全相同（因果 mask 隔离 + 交叉注意力单向）但被重复计算了 50 次。蒸馏到 4 步（DMD2）把这个问题从 50→4 压缩了 12.5 倍，但仍未根除。

从推理基准来看，Cosmos3-Edge I2V 480p 121帧在 H100 上需要约 13 秒——远超过机器人控制的实时性要求（10-100ms）。

### 5.2 解决方案

三频解耦后：

| 频率层 | 模型规模 | 每步推理 | 等效频率 | 适用场景 |
|--------|---------|----------|---------|----------|
| AR 低频 | 8B VLM（prefill 一次） | 1-2 秒 | ~0.5-1 Hz | 任务规划、偏差诊断 |
| Video 中频 | ~5B params（可选触发） | 200-500ms | ~2-5 Hz | 视觉验证、imagination |
| Action 高频 | ~300M params（连续） | 20-50ms | ~20-50 Hz | 实时控制 |

高频 Action 的等效控制频率 = action_chunk_size / 推理延迟。例如：32 步 chunk，推理 1 秒，等效 32 Hz——满足大多数机器人控制需求。

关键机制与 FastWAM 的 `prefill_video_cache` + `forward_action_with_video_cache` 一致：AR K/V 和 Video K/V 缓存后不重算，只跑 Actions DiT 的轻量 FFN。

### 5.3 蒸馏的协同收益

Cosmos 3 的 DMD2 蒸馏（50 步 → 4 步）不仅加速了 Generator FFN，也"顺便"减少了 AR FFN 的冗余计算——蒸馏后 AR FFN 只跑 4 次而非 50 次。在三频架构中，蒸馏主要作用于：
- Video 中频：去噪步数从 50→10-20（因为不需要像素级保真度，latent 空间即可）
- Action 高频：去噪步数从 50→4-8（极少量，或蒸馏到 4 步以下）

---

## 6. 收益二：Test-Time Scaling 动态纠错

### 6.1 问题

即使是训练好的 Action Policy，在部署时仍然会犯错（分布偏移、感知噪声、物理不确定性）。传统的 test-time 做法是 ensemble（多投一次取平均）或多步规划（MPC），但这些方法缺乏对"错误类型"的语义理解。

### 6.2 解决方案

三频架构天然支持语义层级的 test-time scaling：

1. **高频层的 test-time scaling**：对同一状态用不同噪声种子多次去噪，取 ensemble 或最高置信度（轻量，不需要上层参与）
2. **中频层的 test-time scaling**：对候选动作序列用 Video rollout 想象未来，AR 看想象视频打分，选最佳（中等开销）
3. **低频层的 test-time scaling**：AR 发现问题后，用 Chain-of-Thought 深度推理原因和修正策略（最高开销，但最精准）

AR 根据情况自动选择 test-time compute 的投入程度：小偏差简单调整，大偏差深度推理。

### 6.3 AR 的语义纠错能力

```
AR 能给出的不仅是 "偏差 0.5°"，而是语义化的诊断:

  "杯子在第 15 步后开始偏移，根因是 gripper 在第 12 步时闭合角度不够，
   导致杯子在抬升阶段 slip。修正方案: (a) gripper 多闭合 15%， 
   (b) 抬升速度降低 30% 以减少惯性力矩。短期可继续当前任务，
   长期应该在 Action 训练中加入多样化的抓取力度数据。"
```

**这不是一个标量 reward 信号——这是一个完整的故障诊断报告。** 它告诉系统"哪里错了"、"为什么错"、"怎么修正"。

---

## 7. 收益三：强化学习使系统自我提升

### 7.1 自然形成的 RL 框架

AR 的闭环监督能力使整个系统天然适配 RL。AR VLM 在此扮演三重角色：
1. **Reward Model**：评估轨迹质量，给出细粒度得分
2. **Critic**：指出具体问题，给出修正建议
3. **Hindsight Relabeler**：把失败轨迹转化为训练数据（用 AR 建议的正确动作替换原始错误动作）

### 7.2 三种 RL 模式

**Mode 1：结果级 RL（最自然，最便宜）**

每次 episode 结束后，AR 审视完整轨迹，输出结构化 critique：成功/失败模式、根因、哪些段做好、reward score。Action DiT 利用这个 critique 做 RL 微调。

**Mode 2：步骤级 RL + World Model Imagination**

在每个状态，生成多个候选动作序列 → Video 中频 rollout 想象未来 → AR 看想象视频打分 → 选最优 → 产生 `(s, a) → score` 偏好数据用于 DPO/RLAIF。

**Mode 3：闭环自我提升（最完整）**

系统在部署中持续收集数据：
- AR 判断 `replan` 的轨迹片段 → 标记为"错误"，AR 提供修正 → 作为 hindsight 训练数据
- AR 判断 `advance_subgoal` 的轨迹片段 → 标记为"成功"，关联子目标 → 作为正向训练数据
- 离线微调 Action DiT 和 Video DiT（可选）

### 7.3 和传统 Robot RL 的对比

| 维度 | 传统 Robot RL | AR 引导的 RL |
|------|---------------|-------------|
| Reward 设计 | 手工设计，每个任务重新调 | AR 自然语言评估，零手工设计 |
| Critic | 需要学 value function，需要大量数据 | AR 本身就是 Critic，不需要额外训练 |
| 探索 | ε-greedy / entropy bonus，盲目试 | AR 建议的候选方向，有语义引导 |
| 样本效率 | 差，需要大量交互 | 极好——失败轨迹也被转化为训练数据（hindsight relabeling） |
| 零样本泛化 | 差，任务特定的 policy | AR 通用理解 → 快速迁移到新任务 |

### 7.4 三条线的协同增强

```
    多频推理 ──→ 产生行为 ──→ 被 AR 评估 ──→ 产生训练数据
        ↑                                       │
        │                                       ↓
        └──── 推理更快 ──── 模型更准 ──── 强化学习
```

**这是一个自增强系统，每一条都在加速另外两条。**

- **RL → 多频推理**：Action 变准 → 预测误差减少 → AR 醒来次数减少 → 延迟降低并稳定
- **RL → Test-Time Scaling**：模型在训练中学会了应对常见错误 → 推理时不需要频繁纠正 → test-time compute 投入减少
- **Test-Time Scaling → RL**：AR 每次纠正都产生新的训练数据 → 同样的错误下次不会再犯
- **多频推理 → RL**：快速推理 → 更多交互 → 更多训练数据 → 更快收敛

本质上是：**test-time compute 在 RL 中被蒸馏进了模型参数。** 推理时不用再想了，因为训练时已经想过了。

---

## 8. 三条收益的协同增强

随着部署时间增长，系统经历三个阶段：

```
Day 1: Test-Time Scaling 主导
  Action 不够准 → AR 频繁醒来纠正 → 高频 test-time compute
  → 延迟抖动大（有时 50ms，有时 2 秒）
  → 但同时产生大量带 AR critique 的训练数据

Day 7: RL 开始生效
  Action 学会了常见错误的修正方案
  → AR 纠正频率下降 60-70%
  → 延迟趋于稳定
  → System 主要在执行中学

Day 30: 接近自主
  Action 固化：标准操作的肌肉记忆
  Video 学准：rollout 和真实高度一致
  → AR 只在任务开始 + 异常情况时介入
  → 推理几乎全是高频 Action，延迟恒定 ~50ms
  → 新任务快速适应（AR 通用理解 + 之前学到的 primitive skills）
```

---

## 9. 实现路线图

### Phase 1：基础拆解与验证
- 从 Cosmos3-Nano checkpoint 导出 AR 塔和 Generator 塔的权重
- 验证 AR 塔独立推理能力（VANTAGE-Bench score 是否与原始 Cosmos 3 一致）
- 实现基础的 MoT mixed-attention 层

### Phase 2：Action DiT 训练
- 构建 Action DiT 网络（参考 FastWAM，hidden=1024, ffn=4096, 36 layers）
- 在 LIBERO / DROID 数据上训练
- 实现 prefill cache + action-only 推理
- Benchmark：延迟、Action Accuracy vs 原始 Cosmos 3 Policy

### Phase 3：三频协同
- 实现 AR 的结构化输出（subgoal chain + next_check timing + deviation assessment）
- 实现 Video 中频的物理合理性门控
- 实现五种触发机制的调度器
- 端到端评估：任务成功率 + 延迟分布

### Phase 4：RL 闭环
- 实现 AR critique → hindsight relabeling 流水线
- 实现结果级 RL 微调
- 实现步骤级 RL + World Model Imagination
- Benchmark：sample efficiency vs traditional RL

### Phase 5：规模化与泛化
- 多具身形态适配（利用 Cosmos 3 固有的 32 种 embodiment 支持）
- 新任务快速迁移（few-shot + AR 零样本理解）
- 边缘部署优化（FP8/NVFP4 量化 + TensorRT-LLM 编译）

---

## 10. 待解决的关键问题

1. **MoT Mixed-Attention 的具体实现**：是保留 Cosmos 3 的共享 self-attention Q/K/V 模式，还是采用 FastWAM 式的完全独立 expert 拼接？前者参数效率更高但耦合紧，后者解耦但总参数量更大。需要做消融实验。

2. **AR 的输出结构化程度**：完全依赖 AR 的自由文本判断（可能不稳定）vs 使用 constrained decoding 保证 JSON 格式 vs 用一个小的 classifier 做 fallback。需要实测 AR 判断的 consistency。

3. **Video 中频的必要性**：在简单任务（如 pick-and-place）中，Video rollout 是否提供超越 Action 自带条件的信息增益？需要做 ablation（有无 Video 中频对成功率和纠错频率的影响）。

4. **RL 的 credit assignment**：AR 的 critique 如何准确地归因到具体哪几步 action 出了问题？回溯窗口大小如何选择？

5. **三频的能耗模型**：在真实硬件上，不同频率组合的实际功耗和 heat 预算是否允许持续运行？是否需要在 AR 唤醒时动态降低 Action 频率以平衡功耗？

6. **安全性**：AR 给出的修正指令如果本身有误（hallucination），如何通过多频架构的冗余性（Video 物理门控 + Action 执行边界）来兜底？

---

## 附录 A：Cosmos3-Nano 权重结构完整索引

```
全局权重:
  embed_tokens.weight        [151936, 4096]  词表 embedding
  vision_encoder.*           (SigLIP2, 27 blocks)
  lm_head.weight             [151936, 4096]  语言模型头
  action_modality_embed      动作模态 embedding
  audio_modality_embed       音频模态 embedding
  action_proj_in.fc.weight   [32, 262144]    32 种 embodiment × [4096, 64]
  action_proj_out.fc.weight  [32, 262144]    同上，反向投影
  audio_proj_in.weight       [4096, 64]      音频投影
  audio_proj_out.weight      [64, 4096]
  proj_in.weight             [4096, 192]     VAE latent → hidden
  proj_out.weight            [192, 4096]     hidden → VAE latent
  norm.weight                [4096]          AR 最终 norm
  norm_moe_gen.weight        [4096]          Generator 最终 norm
  time_embedder.linear_1     [4096, 256]     时间步投影 1
  time_embedder.linear_2     [4096, 4096]    时间步投影 2

每层权重 (36 layers, 22 tensors/layer):
  self_attn.to_q             [4096, 4096]    共享 Q 投影
  self_attn.to_k             [1024, 4096]    共享 K 投影 (GQA, 8 KV heads)
  self_attn.to_v             [1024, 4096]    共享 V 投影
  self_attn.to_out           [4096, 4096]    共享输出投影
  self_attn.add_q_proj       [4096, 4096]    交叉注意力 Q 投影
  self_attn.add_k_proj       [1024, 4096]    交叉注意力 K 投影
  self_attn.add_v_proj       [1024, 4096]    交叉注意力 V 投影
  self_attn.to_add_out       [4096, 4096]    交叉注意力输出投影
  self_attn.norm_q/ norm_k / norm_added_q / norm_added_k  [128]
  mlp.gate_proj              [12288, 4096]   AR FFN 门控
  mlp.up_proj                [12288, 4096]   AR FFN 上投影
  mlp.down_proj              [4096, 12288]   AR FFN 下投影
  mlp_moe_gen.gate_proj      [12288, 4096]   Generator FFN 门控
  mlp_moe_gen.up_proj        [12288, 4096]   Generator FFN 上投影
  mlp_moe_gen.down_proj      [4096, 12288]   Generator FFN 下投影
  input_layernorm / post_attention_layernorm             [4096]
  input_layernorm_moe_gen / post_attention_layernorm_moe_gen [4096]
```

## 附录 B：FastWAM 关键配置参考

```yaml
# FastWAM 双专家配置
video_dit_config:
  patch_size: [1, 2, 2]
  in_dim: 48
  hidden_dim: 3072      # 大: 视频需要高保真
  ffn_dim: 14336
  text_dim: 4096         # T5 embedding dim
  num_heads: 24
  attn_head_dim: 128
  num_layers: 30

action_dit_config:
  hidden_dim: 1024       # 小: 动作不需要高保真
  ffn_dim: 4096
  text_dim: 4096
  num_heads: 24          # 必须与 Video Expert 一致 (MoT 约束)
  attn_head_dim: 128
  num_layers: 30         # 必须与 Video Expert 一致 (MoT 约束)

# 两个独立的 scheduler（允许不同频率）
video_scheduler:
  train_shift: 5.0
  infer_shift: 5.0
  num_train_timesteps: 1000

action_scheduler:
  train_shift: 5.0
  infer_shift: 5.0
  num_train_timesteps: 1000
```
