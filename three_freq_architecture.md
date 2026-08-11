# Cosmos 3 三频解耦架构：多频推理、Test-Time Scaling 与强化学习的统一框架

> 基于对 NVIDIA Cosmos 3 MoT（Mixture-of-Transformers）架构的深入分析，结合 FastWAM 的多频推理思路，提出将统一模型按"认知频率"拆分并协同运作的研究方向。核心贡献在于发现：将模型拆分为 AR(低频)、DiT Video(中频)、DiT Action(高频)三部分后，可一次性获得三个互相增强的收益——多频推理解决时延、Test-Time Scaling 动态纠错、强化学习使系统自我提升。

---

## 目录

1. [背景：Cosmos 3 MoT 架构深度分析](#1-背景cosmos-3-mot-架构深度分析)
2. [FastWAM 的启示：多频推理的可行性](#2-fastwam-的启示多频推理的可行性)
3. [三频解耦架构设计](#3-三频解耦架构设计)
4. [闭环监督：AR VLM 作为系统观察者](#4-闭环监督ar-vlm-作为系统观察者)
5. [Gate 的实现：规则函数还是可训练参数？](#5-gate-的实现规则函数还是可训练参数)
6. [收益一：多频推理解决时延问题](#6-收益一多频推理解决时延问题)
7. [收益二：Test-Time Scaling 动态纠错](#7-收益二test-time-scaling-动态纠错)
8. [收益三：强化学习使系统自我提升](#8-收益三强化学习使系统自我提升)
9. [三条收益的协同增强：从频繁纠错到自主执行](#9-三条收益的协同增强从频繁纠错到自主执行)
10. [实现路线图](#10-实现路线图)
11. [待解决的关键问题](#11-待解决的关键问题)

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

**FFN 占总参数量的 78.2%**，attention 占 21.8%。这意味着推理的主要计算瓶颈在 FFN——而这也是三频解耦能大幅加速的根本原因。

### 1.4 Self-Attention 与 Cross-Attention 的关系

这是理解 Cosmos 3 架构最核心的一个问题。Self-Attention 的 Q/K/V 来自"所有 token 拼接"，DM tokens 天然可以在全注意力 mask 下 attend AR tokens。那为什么还需要额外的 Cross-Attention？

**因为同一套 Q/K/V 要同时服务三个目标**：AR↔AR 语言建模、DM↔DM 视频去噪、DM→AR 跨模态条件。三个训练任务的梯度在共享的投影矩阵中互相竞争——语言建模希望 K 更好地表示语义，去噪希望 K 更好地表示视频时空结构，跨模态条件希望同一个 K 同时做好两件事。

**Cross-Attention 用另一套同等大小的 Q/K/V/O（每层 42M）专做 DM→AR 的跨模态注入**，解耦了梯度路径。Self-Attention 的共享 Q/K/V 可以专注在单一模态内部的 attention 模式，Cross-Attention 的 add_* Q/K/V 只被跨模态条件一个目标训练。

从机制上看，两者完全不同：Self-Attention 是"token 级混合"（Q/K/V 都来自同一个拼接序列），Cross-Attention 是"Encoder-Decoder 模式"（Q 仅来自 DM tokens，K/V 仅来自 AR tokens）。功能分工上：Self-Attention 负责全局上下文建模和时空一致性，Cross-Attention 负责条件跟随精度——"DM 拿着问题清单专访 AR 专家"。

### 1.5 训练中的 AR↔DiT 相互受益关系

**AR → DiT（直接且巨大）**：
- 语言理解：AR 训练教会的语义通过交叉注意力直接注入 DM token
- 视觉理解：AR 学会的物体边界、材质、空间关系通过 self-attention 中的 DM→AR 通路被 DM 读取
- 物理常识：共享的 self-attention Q/K/V 编码了物理世界规律
- 时序因果：AR 因果注意力训练的能力使 DM 生成的序列有正确的顺序依赖

**DiT → AR（间接且微妙）**：
- AR 不能 attend DM tokens（因果 mask 阻止 + 交叉注意力单向），因此推理时无直接帮助
- 仅在训练过程中通过共享 Q/K/V 的梯度协同传递——视频动态表征和细粒度视觉特征通过共享投影间接提升 AR 的视觉理解

这种不对称是架构的刻意设计：**Generator 被 Reasoner 充分条件，但 Reasoner 不受 Generator 扰动。**

### 1.6 关于消融证据的诚实评估

Paper 公开版（arXiv:2606.02800）**没有**提供教科书式的对照消融表（仅 DiT vs 完整模型、去掉交叉注意力 vs 保留、随机初始化 vs VLM 初始化等）。NVIDIA 用基准横扫（同一 checkpoint 在 VANTAGE-Bench、PAI-Bench、RoboLab、Physics-IQ 同时登顶）作为"存在性证明"。训练流程设计（`freeze_und: false` 默认配置、从 VLM 初始化的决策、交叉注意力 42M/层的参数预算）和 DuoNeural 社区意外消融（修改 AR 通路晚期层权重直接导致生成内容质量下降）提供了间接但有力的证据。

---

## 2. FastWAM 的启示：多频推理的可行性

### 2.1 FastWAM 简介

FastWAM（"Do World Action Models Need Test-time Future Imagination?"，Yuan et al., arXiv:2603.16666）是一个面向机器人操作的世界-动作模型。其核心贡献是证明了**"慢速视频生成 + 快速动作生成"的双频架构是可行的**，且动作推理可以在不重新计算视频通路的情况下完成。

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
              每层: concat([Q_v, Q_a]) attend concat([K_v, K_a], [V_v, V_a])
              然后各自跑 post-block (FFN + cross-attn)
```

### 2.3 FastWAM 的多频调度：核心机制

FastWAM 提供两种推理模式：

**infer_joint（联合推理）**：Video + Action 一起去噪，生成完整视频 rollout + 动作序列。用于训练和评估。

**infer_action（仅动作推理）——这是多频架构的核心**：
1. **Prefill 阶段**（触发一次，~1-2 秒）：将当前帧通过 Video Expert 编码为 latent，跑完 30 层，缓存每层的 K/V
2. **Action 去噪阶段**（循环 N 步，每步 ~50ms）：Action Expert 通过 MoT mixed-attention 读取 Video K/V cache，只跑 Action Expert 的轻量 FFN 和 cross-attention
3. Video Expert 不重算 → 每次去噪只跑 ~300M 参数 → 延迟从秒级降到百毫秒级

**这是 FastWAM 能实现实时机器人控制的核心机制，也是三频架构的直接灵感来源。**

### 2.4 FastWAM MoT vs Cosmos 3 MoT：架构差异

| 维度 | Cosmos 3 | FastWAM |
|------|----------|---------|
| Self-Attention | 1 套共享 Q/K/V/O | N 套独立 Q/K/V/O（每 expert 一套） |
| 参数组织 | Token 级路由（AR token → mlp, DM token → mlp_moe_gen） | Expert 级路由（整个 block 都是独立的） |
| 交叉注意力 | `add_*` 投影，DM→AR 单向，专做条件注入 | 每 expert 内部的 cross-attn，对 text context |
| Attention Mask | AR: causal; DM: full + 额外 cross-attn | per-expert mask: first_frame_causal / group_diagonal |
| 多频推理 | 不支持（联合推理） | 支持（prefill cache + action-only 推理） |
| 设计动机 | 训练侧优化（统一表征） | 推理侧优化（多频调度） |

### 2.5 FastWAM 的局限：文本编码器是瓶颈

FastWAM 使用 T5 作为文本编码器。T5 是通用文本编码器，缺乏物理推理、视觉理解和指令跟随的深层能力。这恰好是 Cosmos 3 AR VLM 的强项——用 Cosmos 3 的 AR 塔替换 T5，是三频架构相比 FastWAM 的关键升级。

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
          ├──────────────┤    ├──────────────┤    ├──────────────┤
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
  3. 在 imagination 中评估多个候选动作序列——类似 Dreamer/MuZero 的 World Model，但是 diffusion-based

**高频 Action DiT（等效 ~30 Hz）—— "执行者"**

- 触发时机：连续运行，异步重规划
- 输入：当前帧（仅第一帧即可）、AR 文本条件、中频 Video latent（可选）
- 去噪步骤：4-8 步（蒸馏后极少量）
- 输出：action chunk [a1, ..., a32]，执行前 8 步后异步重推理
- 等效控制频率 = action_chunk_size / 推理延迟 ≈ 32 / 1s ≈ 32 Hz

### 3.3 同步协议：基于信息差而非时钟

三个频率不是按固定时间间隔触发，而是基于"信息差"——当某一层产生意外信息时向上抛，当某一层做出新决策时向下推。这是"伪同步"模式：各层不是同时开会，而是接力跑。

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

### 3.4 完整的端到端实例

以"把蓝色杯子放到红色架子上"为例，展示三频协同的全过程：

```
t=0    低频: AR 看图像 "蓝色杯子在左边桌子, 红色架子在右边"
             → 子目标链: approach → grasp → lift → translate → place
             → 注入 Video + Action 上下文

t=0.1  中频: Video rollout 验证 "平移路径是否可行"
             → 生成 32 帧: 看起来没问题, 无碰撞
             → Gate: PASS

t=0.2  高频: Action 去噪 4 步 → chunk [a1..a32]
              开始执行 a1..a8

t=0.5  高频: 执行到 a13 时, 预测误差开始累积
             → 关节实际位置和 Diffusion 预测偏差了 0.5° (连续 3 步)
             → 触发条件: "中等偏差, 请求 Video 检查"

t=0.6  中频: Video 用当前真实帧重新 rollout
             → 发现: 杯子稍微偏离了预期轨迹, 但仍然安全
             → 更新 Video K/V cache 供 Action 继续用
             → Gate: PASS (小偏差, 可接受)

t=1.0  高频: Action 继续执行, 进入子目标 3 (translate)
             → 检测到 "gripper 当前 x 坐标已经超过 shelf 位置的 80%"
             → 触发条件: "子目标可能已完成"

t=1.1  中频: Video rollout 验证
             → 确认杯子在 shelf 正上方
             → 触发 AR: "subgoal_3 完成, 进入 subgoal_4"

t=1.3  低频: AR 看最新图像, 确认位置正确
             → "杯子位置 OK. 子目标 4: 放下杯子。注意: 架子表面不平,
               放置角度需要微调 15° 以避免倾倒。"
             → 注入新上下文到 Video + Action

t=1.4  中频: Video rollout 验证 "放下杯子"
             → 发现: 按当前姿态放下, 杯底会先碰到架子边缘
             → 这被物理检测捕获 → 回传给 AR

t=1.5  低频: AR "架子边缘比预期高 3mm, 调整: 抬升手臂 5mm 后再放置"
             → 更新子目标 4

t=1.6  高频: Action 用新条件生成新 chunk
             → 成功放置, 开爪, 完成
```

### 3.5 从 Cosmos 3 权重出发的具体实现步骤

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

### 4.1 核心洞察：Gate 消失

在讨论三频如何协同时，一个自然的想法是设计一个外挂 Gate 模块来决定"AR 什么时候该醒来、什么时候该重规划"。我们深入讨论了 Gate 应该是什么形式——规则函数？可训练的分类器？不确定性检测器？

但当我们追问"AR 能做什么"时，一个关键的洞见浮现了：

**AR 本身就是一个 8B 参数的 VLM。它能看图像、能理解物理世界、能推理因果关系。为什么还需要一个几百K的小分类器来告诉它"你该重规划了"？AR 是房间里最聪明的东西——让它自己决定。**

```
之前的设计:
  Gate (外部) → 决定什么时候触发 AR
  Gate (外部) → 分类器判断物理合理性
  Gate (外部) → 不确定性检测
  Gate (外部) → 子目标完成检测
  → 4 个外部模块, 几百K 参数需要训练

之后的设计:
  AR 看图像 → 自动判断一切
  → 0 个外部模块, 利用已有 8B 参数
```

唯一的"外挂"是超时唤醒（如果 AR 指定的 `next_check_after_steps` 到了但 Action 还在执行，强制唤醒 AR）。这是一个纯规则，不需要训练。

这引出了一个更深层的洞见：**AR VLM 的闭环监督能力本质上就是一个 general-purpose, language-guided, self-correcting reward model。**

### 4.2 AR 的结构化输出

AR 不仅产出"做什么"，还产出"什么时候再来看"和"怎么评估执行效果"。每次 AR 醒来时，它做三件事：

1. **看**：观察当前图像（执行效果）
2. **比**：和上次规划时的预期做比较
3. **判**：决定是否继续、调整还是重规划，以及下次什么时候醒来

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

AR 看到不同情况时，自然地产生不同粒度的响应——这正是 test-time scaling 的本质：

| 情况 | AR 判断 | 响应 | 对下层的影响 | 类比 |
|------|---------|------|-------------|------|
| 一切正常 | `decision: continue` | 设定下一次检查时机 | Action 继续 | "看着没问题, 继续" |
| 小偏差 | `decision: adjust` | 轻量修正指令 | Action 调整方向 | "偏了点, 往左靠靠" |
| 子目标完成 | `decision: advance_subgoal` | 推进到下一子目标 | Action 切换模式 | "抓到了, 可以抬起来了" |
| 显著偏差 | `decision: replan` | 完整重规划 | Action 重新开始 | "杯子掉了! 重新来" |
| 不可恢复 | `decision: abort` | 终止任务 | 全部停止 | "杯子碎了, 放弃" |

这与人类操作者的行为模式完全一致——正常时放手、小偏差轻调、大问题介入。

### 4.4 和人工 Gate 的详细对比

| 维度 | 外挂 Gate | AR VLM 自我监督 | 为什么 AR 更优 |
|------|-----------|-----------------|---------------|
| 预测误差检测 | 手动标定阈值，每任务重新调 | AR 看图像直接判断 | AR 理解"杯子偏移"的语义，不只是数值 |
| 物理合理性 | 训练分类器（~500K params） | AR 自然推理（已具备物理常识） | AR 知道杯子不能悬浮、手臂不能穿墙 |
| 子目标完成 | 训练 detector | AR 看图像 → 比较预期 | AR 理解"在 shelf 上方"意味着什么 |
| 不确定性估计 | Action 附带 uncertainty head | AR 综合判断 | AR 能看到 Action 看不到的上下文 |
| 重规划时机 | 规则函数 | AR 在自己的 think 里决定 | AR 理解任务的语义和优先级 |
| 跨任务泛化 | 每个任务重新设计 | AR 的通用理解→零样本迁移 | "放到架子上"和"放到抽屉里"对 AR 是相似的任务 |

### 4.5 Action 给 AR 的可选摘要

Action 不是必需的汇报对象——AR 可以直接看图像判断一切。但提供一个轻量摘要可以减少 AR 推理的 token 消耗，让 AR 在"看图像"之前有一些上下文：

```python
class ActionStatusSummary:
    """
    Action 高频层给 AR 的可选上下文。
    不替代 AR 的视觉判断——只是让 AR 在有上下文的情况下开始思考，
    类似人类操作者在查看监控画面之前先看一眼日志。
    """
    def summarize(self, chunk_history):
        return f"""
最近一批 {len(chunk_history)} 个 action chunk 的执行情况:
- 动作序列正常, 无急停
- 关节力反馈正常 (无异常阻力)
- gripper 状态: 闭合, 力度正常
- 执行耗时: 1.2s (chunk 0.4s + simulation 0.8s)
        """.strip()
```

### 4.6 完整的执行循环

```python
class ThreeFreqController:
    """
    三频控制器。
    关键设计: AR 不是被定时器唤醒的——它自己决定自己什么时候醒来。
              唯一的"外挂"是超时机制 (当 AR 指定的 steps 到了但 AR 还没醒)。
    """
    
    def __init__(self):
        self.ar = ARSupervisor()       # 低频, 从 Cosmos 3 AR 塔初始化
        self.video = VideoGenerator()  # 中频, 从 Cosmos 3 Generator 塔初始化
        self.action = ActionGenerator() # 高频, 新建 ~300M params
    
    def run(self, task, env):
        frame = env.reset()
        
        # 初始 AR 规划
        ar_plan = self.ar.reason(frame, task)
        video_cache = None
        ar_kv_cache = self.ar.prefill_cache(frame)
        
        while not done:
            # ===== 中频 Video (按需) =====
            if self._should_video_check(ar_plan):
                video_latents, video_cache = self.video.rollout(
                    frame, ar_plan, ar_kv_cache
                )
                if not self._physically_plausible(video_latents):
                    # 物理异常 → 触发 AR 重规划
                    ar_plan = self.ar.replan(frame, task, 
                        reason="video rollout implausible")
            
            # ===== 高频 Action =====
            action_chunk = self.action.infer(
                frame, ar_plan, ar_kv_cache, video_cache
            )
            
            # 执行 action chunk (前 8 步)
            for i in range(0, len(action_chunk), 8):
                for j in range(i, min(i+8, len(action_chunk))):
                    frame, done = env.step(action_chunk[j])
                
                # ===== 低频 AR (按 AR 自己指定的时机醒来) =====
                if self._ar_should_wake(ar_plan):
                    assessment = self.ar.supervise(
                        image=frame,
                        task=task,
                        memory={
                            'prev_expected': ar_plan.expected_scene,
                            'current_subgoal': ar_plan.current_subgoal,
                            'steps_since_last_plan': self.steps_since_last_plan
                        }
                    )
                    
                    if assessment.decision == 'replan':
                        ar_plan = self.ar.replan(frame, task, assessment)
                        break  # 重新开始 action chunk
                    elif assessment.decision == 'adjust':
                        ar_plan.adjustment = assessment.if_adjust
                    elif assessment.decision == 'advance_subgoal':
                        ar_plan = self.ar.advance(assessment)
                        break
                    elif assessment.decision == 'abort':
                        return 'aborted', assessment
                    
                    # AR 更新自己的唤醒时间
                    ar_plan.next_check = assessment.next_check.after_steps
    
    def _ar_should_wake(self, ar_plan):
        # 只有两个条件:
        # 1. AR 自己指定的 steps 到了
        # 2. 硬件异常 (碰撞检测等)
        return (self.steps_since_last_check >= ar_plan.next_check_after_steps
                or self._hardware_anomaly_detected())
```

---

## 5. Gate 的实现：规则函数还是可训练参数？

### 5.1 原则：模型自己能判断的，不要外挂 Gate

在深入讨论 Gate 的实现时，我们发现一个关键的分类法则。Gate 的信号可以分为三类，每一类的实现方式根本不同：

```
信号类型          来源              可微?    最佳实现
─────────────────────────────────────────────────
预测误差          Action vs 观测      ✗ 外部   规则函数
物理检测          Video rollout       ✗ 外部   规则函数 + 轻量分类器
不确定性          Action 内部         ✓ 内部   模型的附带输出
子目标完成        AR 内部            ✓ 内部   AR 自己判断
重规划时机        AR 内部            ✓ 内部   AR 自己判断
```

**模型内部能判断的，不要外挂 Gate。** 外部 Gate 只处理模型"不知道自己不知道"的事情（如预测误差的客观数值、物理碰撞的硬约束）。

### 5.2 外部 Gate：仅处理不可微信号

**预测误差 Gate（纯规则，手动标定）**：这是最简单的 Gate。跑一个 episode，记录正常偏差范围，设 med = 2×正常偏差，high = 5×正常偏差。不需要训练。

**物理合理性 Gate（规则 + 轻量分类器）**：规则部分做粗筛（关节限制、碰撞检测），不需要训练。轻量 MLP 分类器（~500K params）做细判——从真实 robot rollouts（正例）和随机扰动动作生成的 rollouts（负例）中自监督训练。

### 5.3 内部 Gate：模型附带输出

**不确定性 Head**：挂在 Action DiT 上面的轻量模块，和 Action 端到端训练。训练目标为 `(uncertainty - |pred_action - gt_action|)^2`，使不确定性自然校准到实际的预测能力上。

**子目标完成和重规划时机**：完全由 AR 在 Chain-of-Thought think 块里自行判断——充分利用 8B VLM 的推理能力。

### 5.4 关键设计决策

如果非要一个"可训练的 Gate"，本质上是一个小 attention 模块：读 Action 输出的中间特征 + Video rollout latent + AR context embedding，输出 `[continue, video_check, ar_replan]`。

**但为什么这个设计可能不是最优的？** 因为它把 AR 的智能浪费了。你有一个能 Chain-of-Thought 的 8B VLM，为什么还要训练一个几百K的小网络来决定"该不该重新思考"？让 AR 自己决定更合理——它能看到全局画面、理解任务语义、评估风险。

**Gate 的智能应该来自被 Gate 连接的各层模型本身，而不是一个独立的新参数。**

---

## 6. 收益一：多频推理解决时延问题

### 6.1 问题

Cosmos 3 原始架构中，AR FFN 和 Generator FFN 在**每次去噪步**都完整执行。AR 通路每步输出完全相同（因果 mask 隔离 + 交叉注意力单向）但被重复计算了 50 次。蒸馏到 4 步（DMD2）把这个问题从 50→4 压缩了 12.5 倍，但仍未根除。

从推理基准来看，Cosmos3-Edge I2V 480p 121帧在 H100 上需要约 13 秒——远超过机器人控制的实时性要求（10-100ms）。

### 6.2 解决方案

三频解耦后：

| 频率层 | 模型规模 | 每步推理 | 等效频率 | 适用场景 |
|--------|---------|----------|---------|----------|
| AR 低频 | 8B VLM（prefill 一次） | 1-2 秒 | ~0.5-1 Hz | 任务规划、偏差诊断 |
| Video 中频 | ~5B params（可选触发） | 200-500ms | ~2-5 Hz | 视觉验证、imagination |
| Action 高频 | ~300M params（连续） | 20-50ms | ~20-50 Hz | 实时控制 |

高频 Action 的等效控制频率 = action_chunk_size / 推理延迟。例如：32 步 chunk，推理 1 秒，等效 32 Hz——满足大多数机器人控制需求。

关键机制与 FastWAM 的 `prefill_video_cache` + `forward_action_with_video_cache` 一致：AR K/V 和 Video K/V 缓存后不重算，只跑 Actions DiT 的轻量 FFN。

---

## 7. 收益二：Test-Time Scaling 动态纠错

### 7.1 问题与解决

即使是训练好的 Action Policy，在部署时仍然会犯错（分布偏移、感知噪声、物理不确定性）。三频架构天然支持语义层级的 test-time scaling：

1. **高频层的 test-time scaling**：对同一状态用不同噪声种子多次去噪，取 ensemble 或最高置信度（轻量，不需要上层参与）
2. **中频层的 test-time scaling**：对候选动作序列用 Video rollout 想象未来，AR 看想象视频打分，选最佳（中等开销）
3. **低频层的 test-time scaling**：AR 发现问题后，用 Chain-of-Thought 深度推理原因和修正策略（最高开销，但最精准）

### 7.2 AR 的语义纠错能力

AR 的优势在于能给出**语义化的故障诊断**而不是标量 reward：

```
AR 能给出的不仅是 "偏差 0.5°"，而是:

  "杯子在第 15 步后开始偏移，根因是 gripper 在第 12 步时闭合角度不够，
   导致杯子在抬升阶段 slip。修正方案: (a) gripper 多闭合 15%，
   (b) 抬升速度降低 30% 以减少惯性力矩。短期可继续当前任务，
   长期应该在 Action 训练中加入多样化的抓取力度数据。"
```

**这不是一个标量 reward 信号——这是一个完整的故障诊断报告。** 它告诉系统"哪里错了"、"为什么错"、"怎么修正"。"短期"vs"长期"的区分尤其有价值——它指导了什么应该立即纠正、什么应该进入训练队列。

### 7.3 多频 Test-Time Scaling 的关键价值

传统方法中，纠错是盲目的（ensembling 不知道"为什么"应该选另一个 seed 的输出）。三频架构中，AR 的语义理解提供了方向性：不是"换一个试试"，而是"在这个具体维度上偏了，往那个方向修正"。这使得 test-time compute 的分配高度精准——小偏差用高频 ensemble（低开销），大偏差用低频深度推理（高开销但精准）。

---

## 8. 收益三：强化学习使系统自我提升

> 这是三频架构最疯狂的部分：AR VLM 的闭环监督能力使整个系统天然适配 RL。更准确地说——**AR 就是 RL 需要的 Reward Model + Critic + Hindsight Relabeler，三者合一。**

### 8.1 自然形成的 RL 框架

AR VLM 在闭环监督中扮演三重角色：
1. **Reward Model**：评估轨迹质量，给出细粒度得分（不仅是成功/失败，而是部分完成程度）
2. **Critic**：指出具体问题，给出修正建议（不仅是"好"或"不好"，而是"哪里不好，为什么不好，怎么改"）
3. **Hindsight Relabeler**：把失败轨迹转化为训练数据（用 AR 建议的正确动作替换原始错误动作）

### 8.2 三种 RL 模式

#### Mode 1：结果级 RL（最自然，最便宜）

每次 episode 结束后，AR 审视完整轨迹，输出结构化 critique：

```json
{
  "success": false,
  "failure_mode": "grasp_instability",
  "root_cause": "在帧 45 时, gripper 闭合角度不够,
                 导致杯子在抬升时滑落",
  "what_should_have_happened": "gripper 应该多闭合 15%,
                 抬升速度应降低 30%",
  "good_segments": [
    "approaching (帧 1-20) 路径选择合理",
    "shelf_detection (帧 30) 准确识别目标位置"
  ],
  "reward": 0.3  // 部分完成 (接近了 shelf 但没放到)
}
```

Action DiT 用这个 critique 做 RL 微调：AR 的 reward 作为 sparse outcome reward；AR 的 root_cause 和 what_should_have_happened 转化为 hindsight experience（HER 风格）——把失败轨迹的 action label 替换为 AR 建议的正确 action。

#### Mode 2：步骤级 RL + World Model Imagination

在每个状态 s_t，利用 Video 中频做 imagination-based planning：

```python
def step_level_rl(state, ar_context):
    """在每个状态, 生成多个候选 action, 用 Video rollout 评估"""
    candidates = []
    
    for i in range(5):  # 生成 5 个候选
        action_i = action_dit.sample(state, different_noise_seed)
        video_i = video_dit.rollout(state, action_i)  # 想象未来
        score_i = ar_vlm.judge(video_i)  # AR 看想象视频, 打分
        
        candidates.append((action_i, score_i, video_i))
    
    best_action = max(candidates, key=lambda x: x[1])
    
    # 产生偏好数据用于 DPO/RLAIF:
    # (state, winner_action, loser_action) → 偏好优化
    self.buffer.add_preference_data(state, candidates)
    
    return best_action
```

这个模式天然产生了 RL 训练数据：`(s, a) → score` 可用于 Q-learning 或 Behavioral Cloning on best samples。

**这就是 World Model RL（Dreamer/MuZero）的自然实现，但 World Model 是 diffusion-based 的，能产生像素级的想象视频。**

#### Mode 3：闭环自我提升（最疯狂的模式）

系统在部署中持续收集数据，部署时间越长，模型越强：

```python
class ClosedLoopSelfImprovement:
    """
    疯狂的自提升系统。
    
    部署时间越长 → 数据越多 → Action 越准 → AR 醒来越少 → 推理越快。
    这是 test-time compute 被蒸馏进模型参数的自然过程。
    """
    
    def deployment_loop(self, task, env):
        buffer_success = []
        buffer_critiqued = []
        
        while True:
            # ===== 正常执行 =====
            trajectory = []
            frame = env.reset()
            ar_plan = self.ar.reason(frame, task)
            
            while not done:
                action_chunk = self.action.sample(frame, ar_plan.context)
                
                for i in range(0, len(action_chunk), 8):
                    next_frame = env.step(action_chunk[i])
                    trajectory.append({
                        'frame': frame,
                        'next_frame': next_frame,
                        'action': action_chunk[i]
                    })
                    frame = next_frame
                
                # AR 定期检查
                assessment = self.ar.supervise(frame, task, ar_plan.memory)
                
                if assessment.decision == 'replan':
                    # === 这是一个"错误"样本 ===
                    # AR 告诉你错在哪、应该怎么做
                    corrected_action = self.apply_ar_correction(
                        trajectory[-N:].actions,
                        assessment.if_replan.correction_hint
                    )
                    
                    # 写入 buffer: (错误状态, 正确动作, 错误原因)
                    buffer_critiqued.append({
                        'state': trajectory[-N].frame,
                        'wrong_action': trajectory[-N:].actions,
                        'correct_action': corrected_action,
                        'critique': assessment.root_cause,
                        'subgoal': ar_plan.current_subgoal
                    })
                    
                    ar_plan = self.ar.replan(frame, task, assessment)
                    continue
                
                if assessment.decision == 'advance_subgoal':
                    # === 这是一个"成功"样本 ===
                    # 子目标成功完成, 回溯: 最后 M 步是成功的
                    buffer_success.append({
                        'state': trajectory[-M].frame,
                        'actions': trajectory[-M:].actions,
                        'subgoal_name': ar_plan.current_subgoal.name
                    })
                    ar_plan = self.ar.advance(assessment)
            
            # ===== 离线微调 =====
            if len(buffer_success) + len(buffer_critiqued) > threshold:
                self.fine_tune(buffer_success, buffer_critiqued)
    
    def fine_tune(self, success_data, critiqued_data):
        """AR 引导的多目标微调"""
        
        # 1. Behavioral Cloning on AR-verified successes
        for seg in success_data:
            loss_bc = mse_loss(
                self.action(seg.state),
                seg.actions
            )
        
        # 2. Hindsight Learning from AR critiques  
        # AR 说: "gripper 应该多闭合 15%"
        # → 用修正后的 action 作为 pseudo-label
        for seg in critiqued_data:
            loss_hindsight = mse_loss(
                self.action(seg.state),
                seg.correct_action  # AR 建议的正确动作
            )
        
        # 3. World Model improvement (Video DiT)
        # 用成功的真实轨迹训练 Video rollout 更准确
        for seg in success_data:
            loss_video = diffusion_loss(
                self.video(seg.state, seg.actions),
                seg.real_future_frames
            )
        
        return loss_bc + loss_hindsight + loss_video
```

### 8.3 强化学习的奖励瓶颈被自然消除

传统 Robot RL 最困难的部分是 Reward Design——手工设计 reward function 需要大量领域知识，每个任务都要重新调参，而且往往是稀疏的。

在三频架构中，这个瓶颈自动消失了：

- **AR VLM 的自然评估**替代了手工 reward function——不仅给出 success/failure，还给出 partial progress（"你到了 85%，杯子快到 shelf 了但还没放上去，reward=0.3"）
- **AR 的语言化 critique**替代了 value function——不仅打分，还解释为什么，给出修正方向
- **子目标链**提供了自然的稠密奖励——每个子目标的完成都是一个正向信号
- **Hindsight relabeling**让失败也有用——AR 说"这里应该这样做"，失败的轨迹立刻变成训练数据

### 8.4 和传统 Robot RL 的详细对比

| 维度 | 传统 Robot RL | AR 引导的 RL |
|------|---------------|-------------|
| Reward 设计 | 手工设计，每个任务重新调 | AR 自然语言评估，零手工设计 |
| Reward 粒度 | 稀疏（成功/失败）或手工稠密 | 自然稠密（子目标进度 + 部分完成度） |
| Critic | 需要学 value function，需要大量数据 | AR 本身就是 Critic，不需要额外训练 |
| 探索策略 | ε-greedy / entropy bonus，盲目试 | AR 建议的候选方向，有语义引导 |
| 失败样本 | 浪费掉 | 转化为 hindsight training data |
| 样本效率 | 差，需要大量交互 | 极高——失败轨迹也被转化利用 |
| 零样本泛化 | 差，task-specific policy | AR 通用理解 → 快速迁移到新任务 |

### 8.5 和 RL 文献的关系——一个自然的多流派统一

三频架构在 RL 文献中的位置非常自然——它同时涵盖了多个大家各自在做的方向：

```
你的架构:                                RL 流派:
─────────                               ────────

AR 看图像 → 评估偏差 → 语言化判断         Reward Model (RM)
  "杯子偏右了 3cm" → 隐式 penalty         但比 RM 强得多:
  "子目标完成"     → 隐式 reward          不仅打分, 还解释为什么
  "继续" / "重规划" → 稀疏信号             还能给出修正方向

Video rollout → 预测未来 32 帧             World Model (Dreamer/MuZero)
  可以"想象"不同动作序列的后果             但它是 diffusion-based 的
  → 在 imagination 里先试                   视频级别的 fidelity

Action DiT → 从条件生成动作序列            Policy (Actor)
  AR 的判断 = reward                       但条件来自 VLM 而不仅仅是 state
  Video rollout = 可选的 imagination        

AR + Video + Action 一起:                 RLHF / Constitutional RL
  AR 不仅打分, 还"批评"                    但 reward 是 online 的, 
  → 自然地形成 critique pipeline           不是 post-hoc 偏好
```

**这本质上是把 RLHF（Reward Model + Preference Learning）从"训练后对齐文本模型"的范式搬到了"实时闭环控制物理系统"的场景。** AR VLM 的在线自然语言判断替代了 RLHF 中的人工标注偏好，Video DiT 的 imagination 替代了 World Model 的 latent 预测，Action DiT 的扩散去噪替代了传统 Policy 的前向网络。整个 RL pipeline 在同一个三频架构中自然形成。

---

## 9. 三条收益的协同增强：从频繁纠错到自主执行

### 9.1 自增强循环

三条收益不是独立的——它们形成了一个正向反馈环：

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

### 9.2 分阶段收敛：Day 1 → Day 7 → Day 30

```
Day 1: Test-Time Scaling 主导
  Action 不够准 → AR 频繁醒来纠正 → 高频 test-time compute
  → 延迟抖动大（有时 50ms，有时 2 秒）
  → 不可预测, 不可部署
  → 但同时产生大量带 AR critique 的训练数据 (免费标注!)

Day 7: RL 开始生效
  Action 学会了常见错误的修正方案
  → AR 纠正频率下降 60-70%
  → 延迟趋于稳定 (~100-200ms)
  → 系统主要在执行中学

Day 30: 接近自主
  Action 固化：标准操作的肌肉记忆
  Video 学准：rollout 和真实高度一致
  → AR 只在任务开始 + 异常情况时介入
  → 推理几乎全是高频 Action，延迟恒定 ~50ms
  → 新任务快速适应（AR 通用理解 + 之前学到的 primitive skills）
```

**本质上是：test-time compute 在 RL 中被蒸馏进了模型参数。** 推理时不用再想了，因为训练时已经想过了。AR 从"不停的救火队员"变成"偶尔来看看的领导"——这才是真正智能的系统应该有的进化轨迹。

---

## 10. 实现路线图

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

## 11. 待解决的关键问题

1. **MoT Mixed-Attention 的具体实现**：是保留 Cosmos 3 的共享 self-attention Q/K/V 模式，还是采用 FastWAM 式的完全独立 expert 拼接？前者参数效率更高但耦合紧，后者解耦但总参数量更大。需要做消融实验。

2. **AR 的输出结构化程度**：完全依赖 AR 的自由文本判断（可能不稳定）vs 使用 constrained decoding 保证 JSON 格式 vs 用一个小的 classifier 做 fallback。需要实测 AR 判断的 consistency。

3. **Video 中频的必要性**：在简单任务（如 pick-and-place）中，Video rollout 是否提供超越 Action 自带条件的信息增益？需要做 ablation（有无 Video 中频对成功率和纠错频率的影响）。

4. **RL 的 credit assignment**：AR 的 critique 如何准确地归因到具体哪几步 action 出了问题？回溯窗口大小如何选择？

5. **三频的能耗模型**：在真实硬件上，不同频率组合的实际功耗和 heat 预算是否允许持续运行？是否需要在 AR 唤醒时动态降低 Action 频率以平衡功耗？

6. **安全性**：AR 给出的修正指令如果本身有误（hallucination），如何通过多频架构的冗余性（Video 物理门控 + Action 执行边界）来兜底？这要求 Video 和 Action 层在面对 AR 错误指令时能够"反驳"——但这又需要额外的安全机制。

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
