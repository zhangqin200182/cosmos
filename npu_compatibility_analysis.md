# Cosmos 3 & FastWAM NPU (Ascend 910) 兼容性分析

> 环境: PyTorch 2.7.1 + torch_npu, 16 × Ascend 910 (64 GiB HBM), HCCL backend
> 目标: 识别所有 CUDA→NPU 的迁移障碍，评估可行性和工作量

---

## 总览

| 组件 | NPU 兼容风险 | 说明 |
|------|:-----------:|------|
| Cosmos 3 - AR 推理 | ⚠️ 中 | 核心是 Transformers forward, 大部分 op 通用 |
| Cosmos 3 - Generator 推理 | ⚠️ 中-高 | VAE decode 是关键风险点 |
| Cosmos 3 - SFT 训练 | ❌ 高 | FSDP/HSDP, 梯度检查点, Flash Attention |
| FastWAM - 推理 | ⚠️ 中 | Flash Attention + VAE |
| FastWAM - 训练 | ❌ 高 | DeepSpeed ZeRO, 自定义 scheduler |

---

## 第一部分：Cosmos 3 兼容性分析

### 1.1 模型加载

```python
from transformers import Cosmos3OmniForConditionalGeneration
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="auto"
)
```

| 风险点 | 严重度 | 分析 | 解决方案 |
|--------|:-----:|------|---------|
| `device_map="auto"` | ⚠️ 中 | HuggingFace Accelerate 的 `device_map` 通过 `torch.cuda.device_count()` 检测 GPU。Ascend NPU 上这个调用返回 0 | 使用 `device_map={"": "npu:0"}` 或 `max_memory` 手动指定，或直接 `.to("npu")` 然后手动管理显存 |
| `torch.bfloat16` | ✅ 低 | torch_npu 2.7 支持 bf16。Ascend 910 原生支持 | 直接使用 |
| `.from_pretrained()` | ✅ 低 | safetensors 加载是纯 CPU 操作，与设备无关 | 无 |
| `Cosmos3OmniForConditionalGeneration` | ❌ 高 | 需要 transformers >= 5.11.0，当前 4.57.1 没有这个类 | **必须升级 transformers** |

### 1.2 Attention 机制

Cosmos 3 的关键 attention 组件：

| 机制 | 代码位置 | CUDA 依赖 | NPU 兼容性 |
|------|---------|-----------|-----------|
| Standard Self-Attention | `to_q/k/v` → `scaled_dot_product_attention` | ❌ 无 (PyTorch 2.x SDPA 是通用 op) | ✅ 低风险。torch_npu 2.7 实现了 SDPA backend |
| Cross-Attention (add_*) | `add_q_proj` → 独立 attention | ❌ 无 | ✅ 同 Standard Attention，只是 Q/K/V 来源不同 |
| 3D mRoPE | 自定义 freqs_cis + rope_apply | ❌ 无（纯数学运算） | ✅ 低风险。复数运算 (torch.polar, view_as_complex) NPU 支持 |
| QK-Norm (`norm_q/norm_k`) | RMSNorm | ❌ 无 | ✅ 通用 op |
| Attention Mask (causal) | `_create_causal_mask` | ❌ 无 (bool tensor) | ✅ 通用 |
| Attention Mask (full) | 全 1 mask | ❌ 无 | ✅ 通用 |
| GQA (Grouped Query Attention) | 32 heads Q, 8 heads K/V | ❌ 无（只是 reshape） | ✅ 纯张量变换 |

**关键：Cosmos 3 没有使用 `flash_attn` 包或 `xformers` 的 Flash Attention。它使用的是 PyTorch 原生的 `F.scaled_dot_product_attention`（从 weight 结构和 config 中推断：没有 `use_flash_attention` 配置项，使用 `sdpa` 作为 attention implementation）。这大幅降低了 NPU 迁移风险，因为 torch_npu 2.7 已经实现了 SDPA 的 NPU backend。**

### 1.3 DiT / Diffusion 组件

| 组件 | 功能 | CUDA 依赖 | NPU 兼容性 |
|------|------|-----------|-----------|
| Flow Matching Scheduler (UniPC) | 扩散去噪调度 | ❌ 无（纯数学） | ✅ 低风险 |
| Time Embedder | `sinusoidal_embedding → linear` | ❌ 无 | ✅ 通用 op |
| `proj_in/out` | VAE latent ↔ hidden | ❌ 无（Linear） | ✅ 通用 op |
| VAE Encoder/Decoder (Wan2.2 VAE) | 像素 ↔ latent | ⚠️ 中等 | ⚠️ 见 §1.4 |
| CFG (Classifier-Free Guidance) | 正/负 prompt 分支 | ❌ 无（两次 forward） | ✅ 需要 `cfg-parallel-size` 时才用到 `torch.cuda.stream` |
| `torch.Generator(device="cuda")` | 随机种子 | ⚠️ 中等 | ⚠️ 需要改为 `device="npu"` |

### 1.4 VAE 兼容性（关键风险点）

Cosmos 3 使用 **Wan2.2 VAE (AutoencoderKLWan)**，这是一个 3D VAE（包含时间压缩）：

```python
# Wan VAE 的核心操作
# 1. 3D Conv: Conv3d → 可能使用 cuDNN backend
# 2. Temporal downsampling: 帧间压缩
# 3. Spatial downsampling: 空间压缩 (8×)
```

| 风险点 | 严重度 | 分析 |
|--------|:-----:|------|
| Conv3D | ⚠️ 中 | torch_npu 实现了 Conv3D，但可能走的是通用路径而非 NPU 优化路径。**不致命，但可能比 CUDA 慢 3-5×。** |
| Temporal padding/truncation | ✅ 低 | 纯 reshape/slice 操作 |
| Tiled VAE | ⚠️ 中 | 如果 VAE 使用 tiled encoding/decoding（分块处理大分辨率），tile 逻辑涉及的循环和切片可能触发 NPU→CPU 同步开销 |
| `torch.compile` | **❌ 高** | 如果 VAE 或 Wan DiT 的代码中使用了 `torch.compile`，NPU 几乎肯定不支持。**需要在加载时全局禁用。** |

**判断**：VAE 的 encode/decode 是最可能出问题的部分，但问题通常是**性能退化**而非**功能不可用**。对于 Phase 0 的验证性实验，这可以接受。

### 1.5 训练组件

| 组件 | 严重度 | 分析 |
|------|:-----:|------|
| FSDP/HSDP (`torch.distributed.fsdp`) | ❌ 高 | FSDP 的 `ShardingStrategy`、`StateDictType`、`auto_wrap_policy` 等 API 在 HCCL backend 上的行为可能与 NCCL 不同。尤其 `SHARD_GRAD_OP` 和 `FULL_SHARD` 策略需要 HCCL all-gather/reduce-scatter 支持 |
| Gradient Checkpointing (`torch.utils.checkpoint`) | ⚠️ 中 | 核心 `checkpoint` 函数是纯 PyTorch，但 re-computation 时的二阶梯度（如果 DiT 有需要）可能出错 |
| AdamW Optimizer | ✅ 低 | 标准 PyTorch op, torch_npu 支持 |
| Mixed Precision (bf16 AMP) | ⚠️ 中 | `torch.amp.autocast("npu")` API 不同，需要将 `"cuda"` 替换为 `"npu"` |
| `torch.cuda.amp.GradScaler` | ✅ 低 | bf16 不需要 GradScaler（bf16 的动态范围足够） |
| Distributed Process Group | ⚠️ 中 | `dist.init_process_group(backend="hccl")` 而非 `"nccl"`。Cosmos 3 使用 PyTorch FSDP，理论上 backend-agnostic |

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

| 组件 | 严重度 | 分析 |
|------|:-----:|------|
| `accelerate` + `DeepSpeed ZeRO` | ❌ 高 | FastWAM 使用 `accelerate launch` + DeepSpeed ZeRO-1/2。DeepSpeed 的 NPU 支持有限，尤其 `deepspeed.ops` 可能不兼容 |
| `Accelerator(mixed_precision="bf16")` | ⚠️ 中 | `accelerate` 0.x 的 `model.to()` 和 `autocast` 上下文假设 CUDA device |
| `gradient_checkpointing` | ⚠️ 中 | 同 Cosmos 3 |
| `torch.Generator` | ⚠️ 中 | 需要 `device="npu"` |

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

## 第五部分：推荐验证顺序

### Step 1: 环境升级 + 烟雾测试 (1 天)

```bash
# 在 dreamzero_train 容器内
pip install --upgrade "transformers>=5.11.0"
pip install "diffusers @ git+https://github.com/huggingface/diffusers.git"

# 烟雾测试 1: 加载 Cosmos3-Nano AR 塔
python3 << 'EOF'
import torch, torch_npu
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    "nvidia/Cosmos3-Nano",
    dtype=torch.bfloat16,
).to("npu")

# 只测 AR 路径: text + image → text
processor = AutoProcessor.from_pretrained("nvidia/Cosmos3-Nano")
messages = [{"role": "user", "content": [
    {"type": "text", "text": "Caption the image in detail."}
]}]
inputs = processor.apply_chat_template(messages, tokenize=True, 
    add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to("npu", torch.bfloat16)

output = model.generate(**inputs, do_sample=False, max_new_tokens=64)
print("AR OK:", processor.decode(output[0][len(inputs.input_ids[0]):], 
    skip_special_tokens=True)[:100])
EOF
```

### Step 2: FastWAM 烟雾测试 (1 天)

```bash
# 在 dreamzero_train 容器内
git clone FastWAM
cd FastWAM

# 只测推理: 加载 Action DiT
python3 << 'EOF'
# 替换所有 torch.cuda.* → torch_npu.npu.*
# 测试 infer_action 延迟
EOF
```

### Step 3: CUDA→NPU 全局替换脚本 (1-2 天)

针对 Cosmos 3 和 FastWAM 的代码做 §3.1 的系统性替换。

### Step 4: 基准测试 (2-3 天)

对比 CUDA (如果有参考数据) 和 NPU 的:
- AR VLM 推理延迟
- FastWAM Action DiT 推理延迟
- VAE encode/decode 延迟

---

## 核心结论

1. **Cosmos 3 和 FastWAM 都没有使用 `flash_attn` 包或 `xformers`**——两者都依赖 PyTorch 原生 `F.scaled_dot_product_attention`，这已在 torch_npu 2.7 实现。这是最大的好消息。

2. **最可能的阻塞点是分布式训练**（FSDP/DeepSpeed on HCCL）。但 Phase 0 的验证只需要单 NPU 推理，可以先绕过。

3. **Transformer 版本要求是硬障碍**。必须升级，且升级后需要验证与 torch_npu 2.7.1 的兼容性。

4. **VAE 是最大的性能风险**（Conv3D 可能走 NPU 通用路径而慢），但不是功能风险。

5. **CUDA Graph 和 Tensor Parallel 是推理优化的风险**，但 Phase 0 不需要。
