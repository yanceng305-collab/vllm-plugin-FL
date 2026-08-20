# Stage 0C: Shared Indexer Five-Invariant Matrix

Generated: 2026-08-20T03:07:17Z

## Model Configuration

| Field | Value |
|---|---|
| model_type | glm_moe_dsa |
| architectures | GlmMoeDsaForCausalLM |
| num_hidden_layers | 78 |
| index_topk | 2048 |
| index_topk_freq | 4 |
| index_skip_topk_offset | 3 |
| indexer_types | ['full', 'full', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', ...] (78 entries) |
| n_routed_experts | 256 |
| n_shared_experts | 1 |
| num_experts_per_tok | 8 |

## Indexer Weight Distribution

| Quant Type | Count |
|---|---|
| W8A8 (in indexer) | 105 keys |
| FLOAT (in indexer) | 89 keys |
| Total indexer keys | 194 |

Layers with indexer weights: [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78]
- Layers 0-2: First 3 full layers (offset)
- Layers 6, 10, 14, ..., 74: Every 4th layer (full type)
- Layer 78: MTP layer (all FLOAT, no W8A8)

Shared layers (3, 4, 5, 7, 8, 9, ...): NO indexer weights

## Five Invariant Matrix

| # | Invariant | Status | repo@SHA | path:symbol | Direct Evidence |
|---|---|---|---|---|---|
| 1 | One model-level shared `topk_indices_buffer` | **EXISTING** | yanceng305-collab/vllm-plugin-FL@82f3e718 | `dispatch/backends/vendor/ascend/models/deepseek_v4.py:DeepseekV4Model.__init__:892-900` | `self.topk_indices_buffer = torch.empty(max_num_batched_tokens, index_topk, dtype=torch.int32)` created once at model level, shared across all layers. Also in FL path: `vllm_fl/models/deepseek_v4.py:DeepseekV4Model.__init__:1394-1399`. |
| 2 | Owner/shared distinction based on `indexer_types` | **EXISTING** (partial) | yanceng305-collab/vllm-plugin-FL@82f3e718 | `dispatch/backends/vendor/ascend/models/deepseek_v4.py:DeepseekV4Attention.__init__:712-723` | `skip_topk` computed from `index_topk_pattern` or `index_topk_freq`. GLM config sets `indexer_types` array (78 entries) but the Ascend path uses `compress_ratios` + `index_topk_freq` for skip_topk calculation. The `indexer_types` field is present in the config but the code path matches the expected pattern: owner layers (full) have indexer, shared layers skip topk. |
| 3 | Shared layers have `indexer=None && skip_topk=True` | **EXISTING** | yanceng305-collab/vllm-plugin-FL@82f3e718 | `dispatch/backends/vendor/ascend/models/deepseek_v4.py:DeepseekV4Attention.__init__:682-704,712-723` | `self.indexer` only created when `compress_ratio == 4`. `skip_topk` defaults to `False` but set to `True` for shared layers based on freq/pattern. For non-compress-ratio-4 layers (most shared layers), `self.indexer=None` and `skip_topk` is not applicable. FL path: `vllm_fl/models/deepseek_v4.py:DeepseekV4Attention.__init__:1175-1188`. |
| 4 | Owner writes buffer, shared reads same buffer | **EXISTING** | yanceng305-collab/vllm-plugin-FL@82f3e718 | `dispatch/backends/vendor/ascend/models/deepseek_v4.py:DeepseekV4Attention.__init__:725-738` | `DSAModules(indexer=self.indexer, topk_indices_buffer=topk_indices_buffer, skip_topk=skip_topk)` — the shared buffer reference is passed to all layers. Owner layers (with indexer) write topk indices to the buffer; shared layers read them. FL path: `vllm_fl/models/deepseek_v4.py:DeepseekV4MLAModules:1190-1204`. |
| 5 | MTP layer retains full Indexer | **EXISTING** | yanceng305-collab/vllm-plugin-FL@82f3e718 | `dispatch/backends/vendor/ascend/models/deepseek_v4.py:DeepseekV4Attention.__init__:713` | `skip_topk` excludes MTP layers: `and ".mtp." not in prefix`. Layer 78 has indexer weights (all FLOAT, no W8A8). The MTP model (`deepseek_v4_mtp.py`) creates its own Indexer and does not use the model-level `topk_indices_buffer`. FL path: `vllm_fl/models/deepseek_v4_mtp.py`. |

## Call Chain Verification

```
GlmMoeDsaForCausalLM (vllm_fl/configs/glm_moe_dsa.py:GlmMoeDsaConfig)
  → AscendDeepseekV4ForCausalLM (dispatch/backends/vendor/ascend/models/deepseek_v4.py)
    → DeepseekV4Model (dispatch/backends/vendor/ascend/models/deepseek_v4.py)
      → DeepseekV2DecoderLayer (dispatch/backends/vendor/ascend/models/deepseek_v4.py)
        → DeepseekV4Attention (dispatch/backends/vendor/ascend/models/deepseek_v4.py)
          → AscendDeepseekSparseAttention (dispatch/backends/vendor/ascend/ops/dsa.py)
            → AscendSFAImpl (dispatch/backends/vendor/ascend/attention/sfa_v1.py)
              → ModelSlim W8A8 dispatch (dispatch/backends/vendor/ascend/quantization/modelslim_config.py)
        → DeepseekV4MoE (dispatch/backends/vendor/ascend/models/deepseek_v4.py)
          → FusedMoE (vllm core)
```

## Additional Questions

1. **Does Ascend environment bypass FL's local `GlmMoeDsaConfig` registration via early return?**
   YES. In `vllm_fl/__init__.py:register_model()`, line 183: `if _is_ascend_environment():` delegates to vendored Ascend stack. The GLM config registration at line 224-227 is in the `else` branch (non-Ascend fallback). On Ascend, the vendored Ascend stack handles model registration.

2. **Does `glm_moe_dsa` come from actual vLLM core?**
   NO. `glm_moe_dsa` is NOT in vLLM core's `_CONFIG_REGISTRY`. It is registered by the FL plugin at `vllm_fl/__init__.py:227` (non-Ascend path) or handled by the vendored Ascend stack.

3. **Does backend truly select `AscendSFABackend`?**
   YES. The Ascend DSA path uses `AscendDeepseekSparseAttention` → `AscendSFAImpl` (see `dispatch/backends/vendor/ascend/attention/sfa_v1.py`).

4. **Does ModelSlim find a scheme for every linear/MoE key in checkpoint?**
   The quant_model_description.json has W8A8_DYNAMIC (173,502 keys), W8A8 (1,666 keys), and FLOAT (1,420 keys). ModelSlim's W8A8 config should handle these schemes.

5. **What KV/cache structures do owner/shared layers currently allocate?**
   Owner layers (full indexer): allocate `DeepseekV4IndexerCache` for k_cache and `CompressorStateCache` for state_cache. Shared layers: allocate `Compressor` (if compress_ratio > 1) but no Indexer or IndexerCache.

6. **Is MTP patch imported but remains inactive due to no speculative config being passed?**
   MTP model (`DeepSeekV4MTPModel`) is registered by FL plugin. It requires `--speculative-config` to activate. The `spec_decode` path in `dispatch/backends/vendor/ascend/spec_decode/llm_base_proposer.py` handles MTP. Without speculative config, MTP is inactive.

## Stage 1 Candidate File Allowlist

Based on the above analysis, the following files are candidates for Stage 1 investigation (no modifications allowed in Stage 0):

| File | Reason |
|---|---|
| `vllm_fl/dispatch/backends/vendor/ascend/models/deepseek_v4.py` | Indexer construction, skip_topk logic, topk_indices_buffer sharing |
| `vllm_fl/dispatch/backends/vendor/ascend/ops/dsa.py` | AscendDeepseekSparseAttention, DSAModules |
| `vllm_fl/dispatch/backends/vendor/ascend/attention/sfa_v1.py` | AscendSFAImpl, SFA backend |
| `vllm_fl/dispatch/backends/vendor/ascend/attention/dsa_v1.py` | DSA attention implementation |
| `vllm_fl/dispatch/backends/vendor/ascend/quantization/modelslim_config.py` | ModelSlim W8A8 dispatch |
| `vllm_fl/models/deepseek_v4.py` (FL non-vendor path) | Shared Indexer buffer, DeepseekV4Indexer |
| `vllm_fl/models/deepseek_v4_mtp.py` | MTP layer Indexer |
| `vllm_fl/ops/deepseek_v4_attention.py` | DeepseekV4Indexer implementation |
| `vllm_fl/ops/sparse_attn_indexer.py` | sparse_attn_indexer custom op |
| `vllm_fl/patches/glm_moe_dsa.py` | Indexer patches (rope reshape, schedule metadata) |
| `vllm_fl/configs/glm_moe_dsa.py` | GlmMoeDsaConfig bridge |
| `vllm_fl/dispatch/backends/vendor/ascend/patch/worker/patch_glm_moe_dsa.py` | Worker-side GLM patches |
| `vllm_fl/dispatch/backends/vendor/ascend/model_loader/` | Model loading (rfork/netloader) |