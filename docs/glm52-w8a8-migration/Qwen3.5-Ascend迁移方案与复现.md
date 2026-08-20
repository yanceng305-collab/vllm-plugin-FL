> **GLM-5.2 Stage 0 参考说明**
>
> 本文件保存此前同事 Qwen3.5 Ascend 迁移/复现记录，供 `tasks/STAGE-0.md` 的 0A-0 环境 bootstrap 使用。
> Stage 0 **只允许参考 Testing 步骤 0-2 的容器/软件环境搭建方式**；不得照搬本文的 TP=2、Qwen 模型启动命令、graph、async scheduling、MTP、FlashComm、multistream、prefix caching 或性能参数。具体执行边界以最新版 `tasks/STAGE-0.md` 为准。

# Add Qwen3.5 family (27B dense / 35B-A3B MoE; bf16 + w8a8) Ascend NPU support

> 本文件用于 PR 描述 / 复现记录。基于分支 `ascend-model-migration`,仅需**两处小补丁**即可让 Qwen3.5
> 家族(`qwen3_5` dense 与 `qwen3_5_moe` MoE,bf16 与 w8a8)在 Ascend 910B 上端到端跑通。

---

### PR Category
Vendor

### PR Type
New Features

---

## Description(迁移思路)

目标:**Qwen3.5 家族**在 Ascend 910B(Atlas 800 A2)经 vllm-plugin-FL(`VLLM_PLUGINS=fl`)运行,覆盖:

| 模型 | `model_type` / 架构 | 形态 | 量化 |
|------|---------------------|------|------|
| Qwen3.5-27B | `qwen3_5` / `Qwen3_5ForConditionalGeneration` | dense | bf16 / w8a8 |
| Qwen3.5-35B-A3B | `qwen3_5_moe` / `Qwen3_5MoeForConditionalGeneration` | MoE(256 专家 top-8,3B 激活) | bf16 / w8a8 |

两者都是**混合注意力**模型:每 4 层中 3 层是 `linear_attention`(**Gated Delta Net / 线性注意力**,mamba 风格
conv-state + ssm-state)、1 层是 `full_attention`;`mtp_num_hidden_layers=1`(MTP 投机解码);多模态(带
vision tower,可纯文本推理)。w8a8 为 msmodelslim 格式(per-channel int8 权重 + 动态 per-token int8 激活,
含 `quant_model_description.json`)。

**本家族几乎不需要“迁移”代码。** 原因有二:

1. **模型类已在上游 vllm。** 容器内 vllm 0.20.2(`/vllm-workspace/vllm`)自带
   `qwen3_5.py` / `qwen3_next.py` / `qwen3_5_mtp.py`,注册表已含
   `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` / `Qwen3_5MTP` 等,
   无需插件额外 `register_model` 或整套模型 vendor。

2. **NPU 栈已在 vendored 后端里且已接线。** `vllm_fl/dispatch/backends/vendor/ascend/`
   是 vllm-ascend `releases/v0.20.2rc` 的逐位拷贝(仅改 import 路径),其中**已包含完整的
   Qwen3.5 / Qwen3-Next NPU 支持**:
   - GDN 线性注意力 NPU 算子:`ops/gdn.py`(`AscendGatedDeltaNetAttention`,OOT CustomOp 替换上游
     `GatedDeltaNetAttention`)、`ops/triton/fla/*`(chunk/chunk_delta_h/solve_tril/wy_fast/cumsum/l2norm…)、
     `ops/triton/mamba/causal_conv1d.py`、`ops/triton/fused_gdn_gating.py`、AscendC 自定义算子
     (`npu_causal_conv1d_custom` / `npu_recurrent_gated_delta_rule` / `chunk_gated_delta_rule_fwd_h` / `chunk_fwd_o`);
   - monkey-patch:`patch/worker/patch_qwen3_5.py`(补丁 `Qwen3_5DecoderLayer.forward` / `Qwen3NextAttention.forward`)、
     `patch_gdn_attn.py`、`patch_qwen3_next_mtp.py`、`patch_mamba_utils.py`,在 `patch/worker/__init__.py`
     的 `if not is_310p():` 分支下**默认导入**(A2/910B 即生效);
   - 混合 KV/state cache:`NPUPlatform.support_hybrid_kv_cache()=True` + `patch_mamba_config`(block 对齐) +
     v1 model runner 的 mamba 段;MTP 由 `patch_speculative_config` 把模型重写为 `qwen3_5_mtp`;
   - w8a8 量化:`quantization/modelslim_config.py` 的 `packed_modules_model_mapping` 已含 `qwen3_5` /
     `qwen3_5_moe` / `qwen3_next` 的专属映射(含 GDN 的 `in_proj_qkvz` / `in_proj_ba` 投影),
     `AscendW8A8DynamicLinearMethod` / `AscendW8A8DynamicFusedMoEMethod` 分别覆盖 dense 线性层与 MoE 专家。

因此本 PR 的实质工作 = **补齐 1 个原生算子缺口 + 1 个 graph 稳定性修复 + 端到端验证(dense/MoE × bf16/w8a8 全矩阵)**。

---

## Changes

> 共两处代码改动(均在 vendored ascend 后端内,Qwen3.5 家族通用,dense 与 MoE 共享)。

### 1. `AscendGemmaRMSNorm` 兜底(补 `npu_gemma_rms_norm` 缺口)

Qwen3.5 使用 **GemmaRMSNorm**(缩放系数为 `1+weight`)。vendored 后端的
`AscendGemmaRMSNorm.forward_oot` 非 residual 分支直接调用专用融合算子
`torch.ops._C_ascend.npu_gemma_rms_norm`,但 **当前 `libfl_ascend_ops.so` 未包含该算子**,
导致 profiling forward 报 `'_OpNamespace' '_C_ascend' object has no attribute 'npu_gemma_rms_norm'`。

修法(`vllm_fl/dispatch/backends/vendor/ascend/ops/layernorm.py`):用标准 `torch_npu.npu_rms_norm`
配 `(1+weight)` 表达 GemmaRMSNorm(与**同函数 residual 分支**完全一致),并保留“算子存在时走融合快路径”:

```python
if enable_custom_op() and hasattr(torch.ops._C_ascend, "npu_gemma_rms_norm"):
    x, _ = torch.ops._C_ascend.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
else:
    x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight, self.variance_epsilon)
```

- **正确性**:`npu_gemma_rms_norm(x,w,eps) ≡ npu_rms_norm(x,1+w,eps)`,逐位等价(residual 分支本就这么做)。
- **性能**:仅影响**非 residual norm**(整网约 1 处),代价为一次 `1+weight` 加法,可忽略。
- **去命名安全**:不引入任何 `vllm_ascend` / 额外 `.so`,无需重编原生库。

> 说明:GDN 全套 AscendC 算子(conv1d / recurrent / chunk_h / chunk_o / moe_gating_top_k_hash)在该 `.so` 中
> **均已存在**(warmup profiling 通过即证明),`npu_gemma_rms_norm` 是唯一缺口。

### 2. Graph 模式 Dynamo 崩溃修复(早期加载语义)

- **现象**:启用 graph 模式(`cudagraph_mode=FULL_DECODE_ONLY`)后,图捕获可能报
  `torch._dynamo.exc.Unsupported: Skip calling torch.compiler.disable()'d function`——
  norm 的 `forward_oot` 在追踪区首次调用 `enable_custom_op()`,其内部懒加载
  `load_ascend_kernels()`(`@torch._dynamo.disable`),Dynamo 无法 trace。
- **根因**:Qwen3.5 是上游模型,其 `__init__` 不会 eager 调用 `enable_custom_op()`;若在编译前
  eager 调一次,使 `_CUSTOM_OP_ENABLED=True` 置位,追踪到 `if _CUSTOM_OP_ENABLED is not None: return`
  守卫时即被当成常量 True 短路、零 graph-break。
- **修法**(`patch/worker/patch_qwen3_5.py`):包装 `Qwen3_5DecoderLayer.__init__`,在模型构造期
  (编译前)eager 调用一次 `enable_custom_op()`(幂等)。修好后图捕获正常(`Graph capturing finished`)。

```python
_qwen3_5_decoder_init = Qwen3_5DecoderLayer.__init__
def _ascend_qwen3_5_decoder_init(self, *args, **kwargs):
    _qwen3_5_decoder_init(self, *args, **kwargs)
    enable_custom_op()
Qwen3_5DecoderLayer.__init__ = _ascend_qwen3_5_decoder_init
```

---

## Testing(复现测试方式)

> 硬件:单台 **Atlas 800 A2(910B,64G × 8)**。下列命令统一用 **TP=2**(27B/35B-A3B 的 w8a8 与 bf16
> 权重在 TP=2 下均可放下,bf16 35B-A3B 约 34GB/卡)。

### 步骤 0:创建 docker(复用 vllm-ascend 镜像,自带 CANN / torch-npu / 依赖)

```bash
docker pull quay.io/ascend/vllm-ascend:v0.20.2rc1

docker run -d --name fl0202 --net=host --ipc=host --privileged \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /path/to/workdir:/path/to/workdir \
  quay.io/ascend/vllm-ascend:v0.20.2rc1 sleep infinity

docker exec -it fl0202 bash
```

镜像内已有版本:`vllm 0.20.2`(editable,`/vllm-workspace/vllm`)、`torch 2.10.0`、`torch_npu 2.10.0`、
`triton_ascend 3.2.1`。

### 步骤 1:卸载容器中的 vllm-ascend(镜像自带,避免与 fl 插件冲突)

```bash
docker exec fl0202 pip uninstall -y vllm-ascend
docker exec fl0202 pip show vllm-ascend   # 应无输出,确认已卸载
```

### 步骤 2:安装 vllm-plugin-FL(editable,取本 PR 分支)

```bash
cd /path/to/workdir
git clone https://github.com/xiemingda-1002/vllm-plugin-FL.git
cd vllm-plugin-FL && git checkout ascend-model-migration
docker exec fl0202 pip install -e /path/to/workdir/vllm-plugin-FL --no-build-isolation --no-deps
docker exec fl0202 python3 -c "import vllm_fl; print('vllm_fl ok')"
```

### 步骤 3:准备模型(任选其一或多个)

| 权重目录 | 形态 / 量化 | 说明 |
|----------|-------------|------|
| `/path/to/workdir/Qwen3.5-35B-A3B` | MoE / bf16 | 14 个 safetensors 分片 |
| `/path/to/workdir/Qwen3.5-35B-A3B-w8a8-mtp` | MoE / w8a8 | msmodelslim,10 分片 + `quant_model_description.json` |
| `/path/to/workdir/Qwen3.5-27B-w8a8-mtp` | dense / w8a8 | msmodelslim,9 分片 + `quant_model_description.json` |

### 步骤 4:运行与验证(单一启动命令,各变体仅差一两处)

所有**环境变量 + 优化参数**合并为一条命令(对齐 vllm-ascend 官方 Qwen3.5 部署文档,去名到 `VLLM_FL_*`
前缀,经 FL 插件路由),**默认即开启全部优化**:graph 模式(普通 `FULL_DECODE_ONLY`)、async-scheduling、
multistream 共享专家重叠、flashcomm1、`enable_cpu_binding`、MTP 投机解码(`qwen3_5_mtp`,
`num_speculative_tokens=3`)。

以 **Qwen3.5-35B-A3B(MoE,bf16)** 为例,在容器内直接运行:

```bash
# Qwen3.5-35B-A3B (MoE, bf16) on Ascend 910B (Atlas 800 A2) via vllm-plugin-FL
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ---- FL 插件核心环境 ----
export VLLM_PLUGINS=fl                       # 仅加载 fl 插件
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# ---- Ascend 运行时环境 ----
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_FL_ENABLE_FLASHCOMM1=1           # flashcomm1(additional_config 里也开)

cd /tmp   # 避免工作目录下的 ./vllm 遮蔽已安装的 vllm 包
exec vllm serve /path/to/workdir/Qwen3.5-35B-A3B \
  --served-model-name qwen35 \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-num-seqs 32 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --port 8901 \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding": true, "enable_flashcomm1": true, "multistream_overlap_shared_expert": true}' \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'
```

**其它变体只需在上面命令上改一两处:**

- **w8a8(35B-A3B 或 27B)**:把模型路径换成对应 w8a8 目录,并**追加 `--quantization ascend`**。
- **dense 27B**:**去掉 `--enable-expert-parallel`**(dense 无专家,加上会报「专家数必须 >0」);
  `multistream_overlap_shared_expert` 对 dense 无效但无害,可留可去。

> offline 验证可用仓库自带的 `examples/qwen3_5_offline_inference.py`(改 `model=` 与 `tensor_parallel_size`)。

serve 启动后用 curl 验证:

```bash
curl -s http://localhost:8901/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen35","prompt":"The capital of France is","max_tokens":40,"temperature":0}'
```

MTP 命中情况(确认投机解码生效):

```bash
curl -s http://localhost:8901/metrics | grep -E "spec_decode_num_(draft_tokens|accepted_tokens)_total"
```

**prefix-caching 单独验证**(在上面命令基础上把 `--no-enable-prefix-caching` 换成
`--enable-prefix-caching --mamba-cache-mode align`,并**去掉 `--speculative-config`** 后启动):

```bash
curl -s http://localhost:8901/metrics | grep prefix_cache      # 读 queries / hits
python3 /path/to/workdir/test_prefix_cache_client.py           # 发同前缀请求并读 /metrics
```

> 首次启动约需 2–3 分钟:权重加载 + **triton-ascend GDN kernel 首批 JIT 编译**(期间日志会出现
> `No available shared memory broadcast block found in 60 seconds` 心跳告警,属正常)。kernel 缓存到磁盘后,
> 复跑引擎初始化降到 ~30s 量级;graph 模式首次还需 ~40–65s 编译 + ~20–30s 图捕获。

---

## 验证结果(dense / MoE × bf16 / w8a8 全矩阵)

所有变体均用步骤 4 的同一条命令(仅差 `--quantization ascend` 与 dense 去 `--enable-expert-parallel`),
graph 模式为普通 `FULL_DECODE_ONLY`,TP=2。

| 模型 | 形态 | 量化 | MTP | 生成正确性 | MTP 接受率 | 吞吐(40 tok 单请求) |
|------|------|------|-----|-----------|-----------|----------------------|
| Qwen3.5-27B | dense | w8a8 | `qwen3_5_mtp`×3 | ✅(`Paris.`、`Pacific Ocean…`) | ~57% | — |
| Qwen3.5-35B-A3B | MoE | bf16 | `qwen3_5_mtp`×3 | ✅ | ~73% | **38.4 tok/s**(1.04s) |
| Qwen3.5-35B-A3B | MoE | w8a8 | `qwen3_5_mtp`×3 | ✅(`Paris.`、`Pacific Ocean…`) | ~79% | — |

- **w8a8 路径已验证**:dense(27B)与 MoE(35B-A3B)的 w8a8 均跑通——GDN 线性注意力 `in_proj`、注意力 qkv、
  dense/MoE 线性层、MTP 在 w8a8 下均正确;MoE 的 w8a8 fused-MoE(256 专家)亦正常。
- **MTP 投机解码**:三个变体均用 `method=qwen3_5_mtp, num_speculative_tokens=3`,三个投机位均有效接受
  (`/metrics` 的 `spec_decode_num_accepted_tokens_total` 随 position 递减)。

### prefix-caching(混合线性注意力模型)

按步骤 4 的「prefix-caching 单独验证」方式启动(关 MTP + `--enable-prefix-caching --mamba-cache-mode align`):

| 场景 | 结果 |
|------|------|
| 相同前缀两条请求(graph 模式) | ✅ 命中,`prefix_cache_hits_total=1024`;冷请求 → 命中后 ~2.5× 提速,输出正确 |

### 去名验证(改动后仍为 0)

```bash
cd /path/to/workdir/vllm-plugin-FL
grep -rniE "vllm[ _-]ascend" --include=*.py vllm_fl/ | wc -l      # 0
git diff --stat                                                    # ops/layernorm.py + patch/worker/patch_qwen3_5.py
```

---

### Checklist
- [x] dense(27B)/ MoE(35B-A3B)× bf16 / w8a8 端到端生成均正确
- [x] MTP 投机解码(`qwen3_5_mtp`×3)三个变体均生效
- [x] prefix-caching 验证通过(关 MTP 时命中,输出正确)
- [x] 无 `vllm-ascend` / `vllm_ascend` 品牌残留(改动未引入)
- [x] 改动最小(两处小补丁),对齐 vllm-ascend 官方 Qwen3.5 配置
