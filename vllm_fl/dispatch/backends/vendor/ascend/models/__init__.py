from vllm import ModelRegistry


def register_model():
    # The Ascend DeepSeek-V4 model classes are exposed through vllm_fl.models
    # (see vllm_fl/models/deepseek_v4_ascend.py); their implementation lives in
    # this Ascend vendor backend.
    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "vllm_fl.models.deepseek_v4_ascend:AscendDeepseekV4ForCausalLM",
    )

    ModelRegistry.register_model(
        "DeepSeekV4MTPModel",
        "vllm_fl.models.deepseek_v4_mtp_ascend:DeepSeekV4MTP",
    )
