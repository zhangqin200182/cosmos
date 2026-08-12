# Cosmos 3 & FastWAM NPU (Ascend 910) 兼容性分析

> 环境: PyTorch 2.7.1 + torch_npu, 16 × Ascend 910 (64 GiB HBM), HCCL backend
> 基线: DreamZero 已在同环境跑通 Wan2.2 训练/推理，提供了完整 NPU 设备抽象层
> 目标: 识别剩余 CUDA→NPU 迁移障碍，评估可行性和工作量

---

## 总览（更新：经 DreamZero 代码审查后）

| 组件 | 原始风险 | 当前风险 | 说明 |
|------|:---:|:---:|------|
| Cosmos 3 - AR 推理 | ⚠️ 中 | ✅ 低 | DreamZero 设备抽象层 + SDPA 已验证 |
| Cosmos 3 - Generator 推理 | ⚠️ 中-高 | ⚠️ 低-中 | add_* 交叉注意 + 3D mRoPE + VAE 需移植 |
| Cosmos 3 - SFT 训练 | ❌ 高 | ⚠️ 中 | FSDP 已由 monkey-patch nccl→hccl 解决 |
| FastWAM - 推理 | ⚠️ 中 | ✅ 低 | SDPA + Wan VAE 已在 DreamZero 验证 |
| FastWAM - 训练 | ❌ 高 | ⚠️ 中 | DeepSpeed 可通过 monkey-patch 绕过 |

### 剩余硬障碍

**只有 1 个**：Transformers/Diffusers 版本升级。当前 4.57.1/0.30.2，需要 >= 5.11.0 / latest main 才能加载 Cosmos 3 checkpoint。

### DreamZero 已消除的主要风险

| 原高风险 | DreamZero 方案 |
|---------|---------------|
| `torch.cuda.*` → `torch.npu.*` 全局替换 | `device.py` 完整设备抽象层 |
| FSDP/DeepSpeed `nccl` → `hccl` | monkey-patch `init_process_group` 静默转换 |
| Flash Attention NPU 不可用 | 统一 SDPA fallback（与 Cosmos 3/FastWAM 路径一致） |
| Wan2.2 DiT/VAE NPU 兼容 | 完整移植已通过生产训练验证 |
| bf16 AMP / DeviceMesh | `AUTOCAST_DEVICE="npu"` + `parallelize()` 已验证 |

### Cosmos 3 特有组件（需移植验证）

| 组件 | 风险 | 说明 |
|------|:---:|------|
| `add_*` 交叉注意力 | ⚠️ 低 | 底层 SDPA + Linear，无自定义 CUDA kernel，移植工作量大但风险可控 |
| 3D mRoPE | ⚠️ 低-中 | `torch.polar` + `view_as_complex` NPU 支持，fp64 性能弱但仅 prefill 一次 |
| 双 FFN 路由 (mlp / mlp_moe_gen) | ✅ 极低 | 纯 Linear + SiLU，NPU 原生支持 |

---

## 第一部分：Cosmos 3 兼容性分析

### 1.1 模型加载

```python
from transformers import Cosmos3OmniForConditionalGeneration
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="auto"
)
```

**`device_map="auto"`** ⚠️ 中 — Accelerate 通过 `torch.cuda.device_count()` 检测 GPU，NPU 上返回 0。
→ 用 `device_map={"": "npu:0"}` 或直接 `.to("npu")` 手动管理显存。

**`torch.bfloat16`** ✅ 低 — torch_npu 2.7 + Ascend 910 原生支持。

**`.from_pretrained()`** ✅ 低 — safetensors 加载是纯 CPU 操作，与设备无关。

**`Cosmos3OmniForConditionalGeneration`** ❌ 高 — 需要 transformers >= 5.11.0，当前 4.57.1 没有这个类。

### 1.2 Attention 机制

Cosmos 3 的关键 attention 组件：

| 机制 | 代码位置 | CUDA 依赖 | NPU 兼容性 |
|------|---------|-----------|-----------|
| Standard Self-Attention | `to_q/k/v` → SDPA | ❌ 无 | ✅ DreamZero 已验证 SDPA NPU backend |
| Cross-Attention (add_*) | `add_q_proj` → 独立 SDPA | ❌ 无 | ⚠️ 需移植，但底层仍是 SDPA+Linear |
| 3D mRoPE | freqs_cis + rope_apply | ❌ 无 | ⚠️ 需验证复数运算 NPU 兼容性 |
| QK-Norm / Attention Mask / GQA | RMSNorm, bool tensor, reshape | ❌ 无 | ✅ 通用 op |

**关键发现：Cosmos 3 使用 PyTorch 原生 `F.scaled_dot_product_attention`（非 `flash_attn`/`xformers`）。DreamZero 在同 NPU 环境下的 attention 回退策略与此完全一致——`gpu_supports_flash_attention()` 在 NPU 上返回 False，自动走 SDPA fallback。此路径已在生产训练中验证。**

### 1.3 DiT / Diffusion 组件

| 组件 | 功能 | CUDA 依赖 | NPU 兼容性 |
|------|------|-----------|-----------|
| Flow Matching Scheduler (UniPC) | 扩散去噪调度 | ❌ 无（纯数学） | ✅ 低风险 |
| Time Embedder | `sinusoidal_embedding → linear` | ❌ 无 | ✅ 通用 op |
| `proj_in/out` | VAE latent ↔ hidden | ❌ 无（Linear） | ✅ 通用 op |
| VAE Encoder/Decoder (Wan2.2 VAE) | 像素 ↔ latent | ⚠️ 中等 | ⚠️ 见 §1.4 |
| CFG (Classifier-Free Guidance) | 正/负 prompt 分支 | ❌ 无（两次 forward） | ✅ 需要 `cfg-parallel-size` 时才用到 `torch.cuda.stream` |
| `torch.Generator(device="cuda")` | 随机种子 | ⚠️ 中等 | ⚠️ 需要改为 `device="npu"` |

### 1.4 VAE 兼容性

Cosmos 3 使用 **Wan2.2 VAE (AutoencoderKLWan)**，一个 3D VAE（包含时间压缩）。

**DreamZero 已将相同 VAE 移植到 NPU**（`wan_video_vae.py`），通过了训练和推理验证。主要风险点：

- **Conv3D** ⚠️ 低：torch_npu 实现了 Conv3D，但可能走通用路径而非 NPU 优化路径，性能可能比 CUDA 慢 3-5×。Phase 0 推理验证只需 VAE encode 一帧图像（不是 decode 121 帧视频），可接受。
- **Tiled VAE** ⚠️ 低：分块编解码的大分辨率处理逻辑可能触发 NPU→CPU 同步开销。DreamZero 已验证此模式。
- **`torch.compile`** ❌：如果 VAE 代码中使用了 `torch.compile`，NPU 不支持。需全局禁用（DreamZero 已处理）。

**判断**：DreamZero 已验证 VAE 在 NPU 上的功能正确性。性能退化对 Phase 0 验证不致命。

### 1.5 训练组件

**FSDP/HSDP** ~~❌ 高~~ → ✅ DreamZero 已解决。Monkey-patch `init_process_group`（`nccl`→`hccl` 静默转换）消除了主要风险。`parallelize(device_mesh)` 模式已生产验证。

**Gradient Checkpointing** ⚠️ 低 — 核心 `checkpoint` 是纯 PyTorch，DreamZero 已验证。

**Mixed Precision (bf16 AMP)** ⚠️ 低 — 需将 `torch.cuda.amp.autocast()` 替换为 `torch.amp.autocast("npu")`。DreamZero 的 `AUTOCAST_DEVICE="npu"` 提供了参考模式。

**bf16 GradScaler** ✅ — 不需要，bf16 动态范围足够。

### 1.6 推理优化组件

| 组件 | 严重度 | 分析 |
|------|:-----:|------|
| CUDA Graph | ❌ 高 | Cosmos 3 PyTorch 推理 benchmark 使用 CUDA Graphs (`torch.cuda.graph`)。NPU 有自己的 graph 机制 (`torch_npu.npu.graph`)，接口不兼容 |
| Tensor Parallel (`--tensor-parallel-size`) | ❌ 高 | 分布式推理需要 HCCL all-reduce，逻辑可能不同 |
| KV Cache (transformers `use_cache=True`) | ✅ 低 | DynamicCache 是纯 PyTorch tensor 操作 |
| `device_map="auto"` 的多 GPU 分片 | ❌ 高 | `accelerate` 的 `infer_auto_device_map` 使用 CUDA 显存检测 |

---

## 第二部分：FastWAM 兼容性分析

### 2.1 Attention 机制

FastWAM 的 mot.py 使用 **SDPA**，不是 Flash Attention 包：

```python
# wan_video_dit.py 中的 flash_attention 实现:
def flash_attention(q, k, v, num_heads, ctx_mask=None, compatibility_mode=True):
    if compatibility_mode:
        # 只用 F.scaled_dot_product_attention!
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        return x
```

| 风险点 | 严重度 | 分析 |
|------|:-----:|------|
| `F.scaled_dot_product_attention` | ✅ 低 | torch_npu 2.7 实现了 SDPA NPU backend |
| MoT Mixed-Attention (cat Q/K/V) | ✅ 低 | 纯 concat 操作 |
| `compatibility_mode=True` | ✅ 低 | 走的是通用 SDPA 路径，不是 CUDA 专属 |

**这是一个好的发现——FastWAM 的 attention 实现非常干净，不依赖 flash_attn 或 xformers。**

### 2.2 RoPE 实现

```python
def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(...))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)
```

| 风险点 | 严重度 | 分析 |
|------|:-----:|------|
| `torch.view_as_complex` | ⚠️ 低-中 | NPU 对复数操作的性能可能不如 CUDA。**需要性能测试，但功能上应该可行。** |
| `torch.float64` 转换 | ⚠️ 低 | Ascend 910 的 fp64 性能较弱（~1/16 fp16）。RoPE 计算中短暂使用 fp64 用于精度，因为 RoPE 只在 prefill 时算一次，**不致命** |
| `rearrange` (einops) | ✅ 低 | 纯 reshape |

### 2.3 VAE (Wan VAE)

与 Cosmos 3 相同的问题（§1.4），但 FastWAM 主要在 eval 或 `infer_joint` 时使用 VAE decode。在 `infer_action` 模式下（三频架构的主要模式），VAE 只在 prefill 阶段编码一帧图像——负担远小于 Cosmos 3 的 Generator（需要 decode 121 帧视频）。

### 2.4 训练组件

**DeepSpeed ZeRO** ~~❌ 高~~ → ⚠️ DreamZero 的 monkey-patch 可消除 `nccl`→`hccl` 的 backend 问题。但 DeepSpeed 的 NPU 支持本身有限（`deepspeed.ops` 可能不兼容）。Phase 0 可用单 NPU DDP 绕过。

**Accelerate** ⚠️ 低 — `Accelerator(mixed_precision="bf16")` 的 NPU 适配，DreamZero 已提供参考实现。

**Gradient Checkpointing** ⚠️ 低 — 同 Cosmos 3，DreamZero 已验证。

### 2.5 Action DiT

Action DiT (~300M, hidden=1024) 是最简单的部分：

| 组件 | 严重度 | 分析 |
|------|:-----:|------|
| `nn.Linear` | ✅ 低 | 标准 op |
| `nn.LayerNorm` | ✅ 低 | 标准 op |
| `nn.SiLU` / `nn.GELU` | ✅ 低 | 标准 op |
| RoPE (1D, 标准) | ✅ 低 | 比 3D mRoPE 更简单 |

---

## 第三部分：通用迁移 Checklist

### 3.1 全局替换（必须做）

```python
# 设备字符串
"cuda"     → "npu"
"cuda:0"   → "npu:0"

# 设备检测
torch.cuda.is_available()      → torch.npu.is_available()
torch.cuda.device_count()      → torch_npu.npu.device_count()
torch.cuda.current_device()    → torch_npu.npu.current_device()

# Stream
torch.cuda.Stream()            → torch_npu.npu.Stream()
torch.cuda.current_stream()    → torch_npu.npu.current_stream()

# AMP
torch.cuda.amp.autocast()      → torch.amp.autocast("npu")  # 或 torch_npu.npu.amp.autocast()
torch.cuda.amp.GradScaler()    → 删除 (bf16 不需要)

# Random
torch.Generator(device="cuda") → torch.Generator(device="npu")

# Memory
torch.cuda.empty_cache()       → torch_npu.npu.empty_cache()
torch.cuda.memory_allocated()  → torch_npu.npu.memory_allocated()
torch.cuda.max_memory_allocated() → torch_npu.npu.max_memory_allocated()

# Event / Synchronization
torch.cuda.synchronize()       → torch_npu.npu.synchronize()

# CUDA Graph
torch.cuda.graph()             → torch_npu.npu.graph()  # API 可能不同
```

### 3.2 分布式训练替换

```python
# Backend
backend="nccl"  → backend="hccl"

# 初始化
torch.distributed.init_process_group(backend="hccl")

# FSDP device
fsdp_device = torch.device("npu")

# All-Reduce / All-Gather
# torch_npu 实现了 HCCL 的 all_reduce / all_gather
# API 与 NCCL 一致 (torch.distributed.all_reduce 等), 
# 但底层通信库不同
```

### 3.3 Accelerate 配置

```yaml
# accelerate config 需要修改:
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config:
    zero_optimization:
      stage: 2
  # DeepSpeed ZeRO on NPU 需要验证

# 或者使用 FSDP 替代 DeepSpeed:
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_state_dict_type: FULL_STATE_DICT
```

### 3.4 禁用的功能

| 功能 | 原因 |
|------|------|
| `torch.compile` | 不支持 Ascend backend |
| `flash_attn` package | CUDA 专属。使用 SDPA 替代 |
| `xformers` | CUDA 专属 |
| `bitsandbytes` 量化 | CUDA 专属 |
| `triton` kernel | CUDA 专属 |
| CUDA Graph (`torch.cuda.graph`) | 需要改为 NPU graph API |
| `device_map="auto"` | 需要手动指定 NPU device |

---

## 第四部分：风险矩阵和优先级

### A 级风险（阻塞性，必须解决）

| 风险 | 影响 | 解决方案 | 预估工时 |
|------|------|---------|---------|
| Transformers 版本 | 无法加载 Cosmos 3 checkpoint | 升级到 >= 5.11.0，验证兼容性 | 2-4h |
| Diffusers 版本 | 无法使用 Cosmos3OmniPipeline | 安装最新 main 分支 | 1-2h |
| `torch.cuda.*` 调用 | 运行时崩溃 | 全局替换脚本 (§3.1) | 4-8h |
| FSDP/DeepSpeed on HCCL | 无法训练 | 先用单 NPU DDP 验证，再逐步上 FSDP | 2-5 天 |
| Mixed Precision autocast | AMP 不工作 | 替换为 `torch.amp.autocast("npu")` | 2-4h |

### B 级风险（性能退化，不阻塞但需要关注）

| 风险 | 影响 | 缓解方案 |
|------|------|---------|
| Conv3D 性能 (VAE) | VAE encode/decode 慢 3-5× | 减小 VAE tile size；编解码非实时路径可接受 |
| RoPE 复数运算 (fp64) | Prefill 慢 2-3× | Prefill 是一次性的，摊到 N 次推理上可忽略 |
| SDPA NPU backend 性能 | Attention 慢 | 与 CUDA 相比的基准测试 |

### C 级风险（可绕过）

| 风险 | 影响 | 绕过方案 |
|------|------|---------|
| CUDA Graph | 推理延迟高 10-20% | 不用 CUDA Graph（Phase 0 不需要极致性能） |
| Tensor Parallel | 无法多 NPU 推理大模型 | Phase 0-2 不需要（Nano 单 NPU 能装下） |
| NPU→CPU 数据拷贝延迟 | 日志/打印慢 | 减少中间步骤的同步 |

---

## 第五部分：推荐验证顺序（更新）

### Step 0: 直接复用 DreamZero 设备抽象层 (0.5 天)

不需要重写 NPU 适配。在 Cosmos 3 / FastWAM 的代码入口处 import DreamZero 的 `device.py`，或将其核心模式（monkey-patch `init_process_group`、SDPA fallback）复制到目标项目中。

```python
# 最简方式：直接设置环境变量让 DreamZero 的设备抽象生效
import os
os.environ["DREAMZERO_DEVICE"] = "npu"
from groot.vla.common.utils.device import DEVICE, DEVICE_TYPE, synchronize, empty_cache
```

### Step 1: 版本升级 + 烟雾测试 (0.5 天)

```bash
pip install --upgrade "transformers>=5.11.0"
pip install "diffusers @ git+https://github.com/huggingface/diffusers.git"

# 烟雾测试: 加载 Cosmos3-Nano AR 塔
python3 -c "from transformers import Cosmos3OmniForConditionalGeneration; print('OK')"
```

### Step 2: AR 推理验证 (1 天)

利用 DreamZero 的设备抽象层 + 升级后的 HF，验证 AR 塔在 NPU 上的推理能力和延迟。

### Step 3: FastWAM Action DiT 推理 (1 天)

复用 DreamZero 已验证的 SDPA + Wan VAE 路径。

### Step 4: 基准测试 (1 天)

AR VLM 推理延迟、Action DiT 推理延迟、VAE encode 延迟 vs CUDA 参考数据。

**总计：约 4 天**（比原始估计缩短 60%，因为 DreamZero 已消化了大部分 CUDA→NPU 移植工作）。

---

## 第六部分：DreamZero 已解决的问题（对比分析）

DreamZero 是一个已经在 Ascend 910 NPU 上跑通的 World Action Model 训练/推理框架。通过分析其代码（`groot/vla/common/utils/device.py`, `modules/attention.py`, `modules/wan2_1_attention.py`, `experiment/base.py` 等），以下是 DreamZero 已经处理的 NPU 兼容性问题的逐项对照：

### ✅ 已解决：设备抽象层（完整覆盖 §3.1 全局替换）

DreamZero 的 `device.py` 是一个**生产级的设备抽象层**，覆盖了我之前列出的所有全局替换项：

| 原分析中的风险 | DreamZero 解决方案 | 状态 |
|---------------|-------------------|:---:|
| `torch.cuda.is_available()` | `device.py:51-52` — `torch.npu.is_available()` 自动检测 | ✅ |
| `torch.cuda.device_count()` | `device.py:166-168` — `torch.npu.device_count()` | ✅ |
| `torch.cuda.Stream()` | `device.py:76` — `_AcceleratorStream = torch.npu.Stream` | ✅ |
| `torch.cuda.Event()` | `device.py:75` — `_AcceleratorEvent = torch.npu.Event` | ✅ |
| `torch.cuda.synchronize()` | `device.py:206-209` — `torch.npu.synchronize()` | ✅ |
| `torch.cuda.empty_cache()` | `device.py:235-238` — `torch.npu.empty_cache()` | ✅ |
| `torch.cuda.memory_allocated()` | `device.py:251-255` — `torch.npu.memory_allocated()` | ✅ |
| `torch.cuda.max_memory_allocated()` | `device.py:261-264` — `torch.npu.max_memory_allocated()` | ✅ |
| `torch.Generator(device="cuda")` | `device.py:DEVICE_STR` — `f"npu:{LOCAL_RANK}"` | ✅ |
| `torch.cuda.manual_seed_all()` | `device.py:316-319` — `torch.npu.manual_seed_all()` | ✅ |
| ProfilerActivity | `device.py:78-80` — `torch_npu.profiler.ProfilerActivity.NPU` | ✅ |

### ✅ 已解决：分布式训练 Backend（覆盖 §3.2）

最巧妙的设计在 `device.py:378-389`：

```python
# Monkey-patch torch.distributed.init_process_group
# 当 backend="nccl" 在 NPU 上被调用时，自动替换为 "hccl"
_original_init_process_group = torch.distributed.init_process_group

def _patched_init_process_group(*args, **kwargs):
    if DEVICE_TYPE == "npu":
        if kwargs.get("backend", "") == "nccl":
            kwargs["backend"] = "hccl"
    return _original_init_process_group(*args, **kwargs)

if DEVICE_TYPE == "npu":
    torch.distributed.init_process_group = _patched_init_process_group
```

这意味着**任何使用 `backend="nccl"` 的第三方库**（HuggingFace Trainer、Accelerate、DeepSpeed）都不需要修改代码——这个 monkey-patch 会静默地将 `nccl` 转换为 `hccl`。**这是一个优雅的解决方案，直接消除了我分析中 FSDP/DeepSpeed on HCCL 的主要迁移风险。**

### ✅ 已解决：Flash Attention / SDPA 回退（覆盖 §1.2, §2.1）

DreamZero 的 `attention.py:88-99` 和 `wan2_1_attention.py:115-125` 实现了完全一致的策略：

```python
# 在 NPU 上: gpu_supports_flash_attention() 返回 False
# → 自动使用 _sdpa_attention_fallback()
# → 调用 torch.nn.functional.scaled_dot_product_attention()
```

`device.py:396-414` 明确注释：

```python
def gpu_supports_flash_attention() -> bool:
    """On NPU: always returns False so SDPA fallback is used.
       torch_npu natively supports F.scaled_dot_product_attention,
       so no custom kernel is needed."""
    if DEVICE_TYPE != "cuda":
        return False
```

**这意味着 DreamZero 在 NPU 上走的是和 Cosmos 3、FastWAM 完全相同的 attention 路径——PyTorch 原生 SDPA。** 这个验证已经完成。

### ✅ 已解决：Wan2.2 DiT / VAE（覆盖 §1.4, §2.3）

DreamZero 已经完整移植了 Wan2.2 的 DiT 和 VAE 到 NPU：

```
dreamzero/groot/vla/model/dreamzero/modules/
├── wan_video_dit.py          ← Wan DiT backbone
├── wan_video_dit_action_casual_chunk.py  ← Action-conditioned DiT
├── wan_video_vae.py          ← Wan VAE (3D Conv encode/decode)
├── wan_video_text_encoder.py ← T5 text encoder
├── wan_video_image_encoder.py ← Image encoder
├── wan2_1_attention.py       ← Attention (FA2/FA3/SDPA/TE multi-backend)
├── wan2_1_submodule.py       ← Sub-modules (RMSNorm, RoPE, etc.)
└── flow_match_scheduler.py   ← Flow Matching scheduler
```

**这一整套组件已经经过了 NPU 训练和推理的验证。** Cosmos 3 在新环境中不需要重写 VAE 或 DiT backbone——可以直接复用 DreamZero 已验证的组件或参考其适配模式。

### ✅ 已解决：FSDP / Device Mesh（覆盖 §3.2）

`base_vla.py:580-581`：

```python
def parallelize(self, device_mesh: DeviceMesh):
    self.action_head.parallelize(device_mesh=device_mesh)
```

DreamZero 使用 PyTorch 原生的 `DeviceMesh` + FSDP，通过 monkey-patch 的 `hccl` backend 运行。这意味着 **FSDP 训练在 NPU 上已经过生产验证**，不需要回退到单 GPU DDP。

### ⚠️ 部分解决：AMP Mixed Precision

`device.py:73` 设置了 `AUTOCAST_DEVICE = "npu"`，但 Cosmos 3 和 FastWAM 的代码中可能直接调用 `torch.cuda.amp.autocast()`。需要替换为 `torch.amp.autocast(DEVICE_TYPE)` 或 DreamZero 风格的 device-agnostic 写法。**模式已知，工作量小。**

### ❌ 仍需解决

1. **Transformers/Diffusers 版本** — 需升级到 >= 5.11.0 才能加载 Cosmos 3 checkpoint。DreamZero 的 `VLA.from_pretrained` 提供了 safetensors 直读模式，可绕过 HF 类直接操作权重。

2. **Cosmos 3 `add_*` 交叉注意力** — DreamZero 使用的是标准 cross-attention（对 text）。Cosmos 3 的 `add_*`（DM→AR 单向注入）是新的 attention 模式，需移植到 NPU，但底层仍是 SDPA，功能上可行。

3. **3D mRoPE** — DreamZero 使用标准 1D RoPE。Cosmos 3 的 3D mRoPE（三组维度独立 position）需验证 NPU 上 `torch.polar` / `view_as_complex` 兼容性。

4. **Cosmos 3 Generator 推理 pipeline** — DreamZero 是 backbone + action_head 架构。Cosmos 3 的 Diffusers `Cosmos3OmniPipeline` 流程不同，Phase 0 验证需要直接加载完整 Generator。

### 对比总结

原分析列出 30+ 风险点，经 DreamZero 代码审查后的状态：

| 解决程度 | 占比 | 覆盖范围 |
|:-------:|:---:|------|
| ✅ 已解决 | ~70% | 设备抽象、分布式 backend、Flash Attn 回退、Wan DiT/VAE、FSDP |
| ⚠️ 部分解决 | ~15% | AMP 语法、device_map |
| ❌ 仍需处理 | ~15% | HF 版本升级、add_* 交叉注意力、3D mRoPE、Generator pipeline |

**关键发现：**

1. DreamZero 的 `device.py` 是一个可以直接复用的设备抽象层
2. Monkey-patch `init_process_group` 消除了分布式训练的最大风险
3. SDPA fallback 策略和 Cosmos 3/FastWAM 的 attention 实现完全一致
4. Wan2.2 全家桶已在 NPU 验证 → Cosmos 3 Generator 的 DiT 和 VAE 部分可参考
5. 真正需要从零适配的是 Cosmos 3 特有的 `add_*` 交叉注意力和 3D mRoPE

---

## 核心结论（最终版）

1. **硬障碍只有 1 个**：Transformers/Diffusers 版本升级。其余问题要么已被 DreamZero 消除，要么是 Cosmos 3 特有组件的移植工作（底层 op 均 NPU 支持）。

2. **DreamZero 已消化 ~70% 的 NPU 迁移工作**：设备抽象层、分布式 backend、SDPA fallback、Wan DiT/VAE、FSDP 全部通过了生产验证。

3. **Cosmos 3 特有移植（add_* 交叉注意 + 3D mRoPE）风险可控**：底层均为标准 PyTorch op（SDPA、Linear、复数运算），无自定义 CUDA kernel。DreamZero 已验证了同类型的 attention 和 RoPE 在 NPU 上的正确性。

4. **单 NPU 推理验证（Phase 0）可立即开始**：只需版本升级 + 复用 DreamZero 设备抽象层。预计 4 天可完成全链路烟雾测试。
