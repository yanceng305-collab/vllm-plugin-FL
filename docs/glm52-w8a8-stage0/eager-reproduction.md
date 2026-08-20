# Stage 0D: Minimal Eager First-Error Reproduction

Generated: 2026-08-20T03:21:43Z

## Command

```bash
export VLLM_PLUGINS=fl
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_USE_V1

vllm serve /data/czh/GLM-5.2-w8a8 \
  --enforce-eager \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --seed 42 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --port 18000
```

## Configuration Summary

| Parameter | Value | Justification |
|---|---|---|
| VLLM_PLUGINS | fl | Required for FL plugin |
| enforce-eager | True | Stage 0: no graph/compilation |
| quantization | ascend | Required for Ascend NPU |
| trust-remote-code | True | Required for GLM MoE DSA architecture |
| max-model-len | 2048 | Minimal to fit memory |
| max-num-seqs | 1 | Minimal batch size |
| seed | 42 | Fixed for reproducibility |
| tensor-parallel-size | 1 | Minimal TP for debug |
| gpu-memory-utilization | 0.90 | Standard |

## Disabled/Omitted (per STAGE-0.md)

- Speculative/MTP: no `--speculative-config`
- Compilation/ACL graph: `--enforce-eager`
- Prefix caching: default (not explicitly disabled)
- Async scheduling: default
- DSA-CP: not enabled
- FlashComm/MC2: not enabled
- Multistream/shared-expert overlap: not enabled
- EPLB: not enabled

## First Root Error

### Location

```
File: vllm_fl/dispatch/backends/vendor/ascend/quantization/modelslim_config.py:330
Function: get_linear_quant_type
Line: quant_type = quant_description[prefix + ".weight"]
```

### Error

```
KeyError: 'model.layers.3.self_attn.indexer.wq_b.weight'
```

### Full Traceback (condensed)

```
EngineCore failed to start.
  File "vllm/v1/engine/core.py", line 1110, in run_engine_core
    engine_core = EngineCoreProc(...)
  ...
  File "vllm_fl/dispatch/backends/vendor/ascend/worker/model_runner_v1.py", line 3390, in load_model
    self.model = get_model(vllm_config=self.vllm_config)
  ...
  File "vllm/model_executor/models/deepseek_v2.py", line 985, in __init__
    self.indexer = Indexer(...)
  File "vllm/model_executor/models/deepseek_v2.py", line 632, in __init__
    self.wq_b = ReplicatedLinear(...)
  File "vllm_fl/dispatch/backends/vendor/ascend/ops/linear.py", line 521, in __init__
    AscendLinearBase.__init__(...)
  File "vllm_fl/dispatch/backends/vendor/ascend/ops/linear.py", line 121, in __init__
    self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
  File "vllm_fl/dispatch/backends/vendor/ascend/quantization/modelslim_config.py", line 543, in get_quant_method
    scheme = create_scheme_for_layer(self.quant_description, prefix, ...)
  File "vllm_fl/dispatch/backends/vendor/ascend/quantization/modelslim_config.py", line 384, in create_scheme_for_layer
    quant_type = get_quant_type_for_layer(quant_description, prefix, ...)
  File "vllm_fl/dispatch/backends/vendor/ascend/quantization/modelslim_config.py", line 363, in get_quant_type_for_layer
    return get_linear_quant_type(quant_description, prefix, ...)
  File "vllm_fl/dispatch/backends/vendor/ascend/quantization/modelslim_config.py", line 330, in get_linear_quant_type
    quant_type = quant_description[prefix + ".weight"]
KeyError: 'model.layers.3.self_attn.indexer.wq_b.weight'
```

## Root Cause Analysis

### Call Chain

```
GlmMoeDsaForCausalLM (vllm core: deepseek_v2.py:1376)
  → DeepseekV2Model.__init__ → make_layers
    → DeepseekV2DecoderLayer.__init__ (layer 3)
      → DeepseekV4Attention.__init__ (Ascend vendored)
        → compress_ratio == 4 → creates Indexer
        → Indexer.__init__ (vllm core: deepseek_v2.py:632)
          → ReplicatedLinear(wq_b) → AscendLinearBase.__init__
            → quant_config.get_quant_method()
              → modelslim_config.py:get_quant_type_for_layer
                → quant_description["model.layers.3.self_attn.indexer.wq_b.weight"]
                  → KEY ERROR
```

### Gap Description

1. **vLLM core's `DeepseekV2DecoderLayer`** creates an `Indexer` for ALL layers where `compress_ratio == 4`, regardless of `indexer_types` configuration.
   - File: `vllm/model_executor/models/deepseek_v2.py:985`
   - `self.indexer = Indexer(...)` is called unconditionally for all compress_ratio=4 layers.

2. **GLM-5.2 checkpoint** only has indexer weights for "full" (owner) layers:
   - Layers with indexer weights: [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78]
   - Layers WITHOUT indexer weights: [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, ...]

3. **`modelslim_config.py`** tries to look up the quantization scheme for every linear layer, including the indexer's `wq_b` linear layer, but fails because the key doesn't exist in `quant_description` (which maps checkpoint weight keys to quantization types).

4. **The `indexer_types` config** (which correctly identifies shared vs owner layers) is available in the model config but is NOT consulted during Indexer construction in the vLLM core path.

### First Failure Location

| Field | Value |
|---|---|
| Layer | 3 |
| Layer Type | shared (indexer_types='shared') |
| File | `vllm_fl/dispatch/backends/vendor/ascend/quantization/modelslim_config.py` |
| Line | 330 |
| Symbol | `get_linear_quant_type` |
| Error | `KeyError: 'model.layers.3.self_attn.indexer.wq_b.weight'` |
| Root Cause | Indexer created for shared layer that has no indexer weights in checkpoint |

### Downstream Noise

The subsequent `Process EngineCore` stack trace printed after the initial error is downstream noise from the worker process failing to start. The root cause is the single `KeyError` at layer 3.

## Complete Log

Saved to: `docs/glm52-w8a8-stage0/logs/node-0/eager-repro-tp1.log`

## Confirmed Facts

1. Model resolves to `GlmMoeDsaForCausalLM` architecture ✅
2. ModelSlim W8A8 quantization is activated: `Using the vllm-plugin-FL modelslim Quantization now!` ✅
3. NPU is detected and initialized ✅
4. HCCL distributed init succeeds (rank 0, world_size=1) ✅
5. Error occurs during model construction (weight loading), not during inference ✅
6. First failure is at layer 3 (first shared layer with compress_ratio=4) ✅

## Inferences

1. The `indexer_types` config is not used by the vLLM core code path to skip Indexer creation for shared layers.
2. The Ascend vendored `DeepseekV4Attention` has `skip_topk` logic but it's not used to prevent Indexer construction.
3. The `GLM-5.1_best_practice.yaml` in the checkpoint directory may contain the correct configuration for enabling IndexCache/skip_topk behavior.

## Unknowns

1. Whether the `GLM-5.1_best_practice.yaml` contains the necessary `use_index_cache` or `index_topk_pattern` settings.
2. Whether the vLLM 0.20.2 core's `DeepseekV2DecoderLayer` is expected to handle this case or if the FL plugin is expected to override the attention class.
3. The full `best_practice.yaml` content and recommended launch configuration.

## Stage 1 Candidate File Allowlist

| File | Reason |
|---|---|
| `vllm_fl/dispatch/backends/vendor/ascend/quantization/modelslim_config.py:330` | Direct error site — needs to handle missing keys gracefully |
| `vllm_fl/dispatch/backends/vendor/ascend/models/deepseek_v4.py:682-704` | Indexer construction should check indexer_types |
| `vllm/model_executor/models/deepseek_v2.py:985` | vLLM core unconditionally creates Indexer for all c4 layers |
| `vllm_fl/dispatch/backends/vendor/ascend/patch/worker/patch_glm_moe_dsa.py:49` | GLM-specific decoder init patch |
| `vllm_fl/patches/glm_moe_dsa.py` | Platform patches for GLM MoE DSA |
| `/data/czh/GLM-5.2-w8a8/GLM-5.1_best_practice.yaml` | Contains recommended config for this model |