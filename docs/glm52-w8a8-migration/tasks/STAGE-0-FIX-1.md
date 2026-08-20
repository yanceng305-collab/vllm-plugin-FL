# Stage 0 补证任务 1：修正 PR #1 的矩阵、复现和 artifact

## 任务身份

- Task ID：`GLM52-W8A8-STAGE-0-FIX-1`
- 执行者：DeepSeek
- 目标 PR：[yanceng305-collab/vllm-plugin-FL#1](https://github.com/yanceng305-collab/vllm-plugin-FL/pull/1)
- 当前验收：`NEEDS-FIX`
- 允许内容：只补充或修正 Stage 0 证据
- 禁止内容：生产源码、控制面、Stage 1 实现、新 PR、合并或 force-push

在当前 `audit/glm52-w8a8-stage0-gap` 分支追加 commit，保持 Draft PR #1；完成后停止并等待复审。

## 1. 重写 shared Indexer 五项矩阵

矩阵必须只描述实际 GLM-5.2/vLLM 0.20.2/FL 调用链，不得用未执行的 DeepSeek-V4 model 路径替代：

```text
GlmMoeDsaForCausalLM (vLLM core deepseek_v2.py)
  -> DeepseekV2Model
  -> DeepseekV2DecoderLayer
  -> DeepseekV2MLAAttention
  -> FL OOT AscendMultiHeadLatentAttention
  -> Ascend SFA selection/implementation
```

至少按以下结论修正，并为每项提供实际 repo@SHA、path:symbol 和直接证据：

1. 模型级 `topk_indices_buffer`：可标 `EXISTING`，但证据应来自 vLLM core `DeepseekV2Model` 的实际路径，不是 DeepSeek-V4 model。
2. 依据 `indexer_types` 区分 owner/shared：`MISSING` 或 `INCOMPATIBLE`；真实日志已证明 layer 3 没有被识别为 shared。
3. shared 层 `indexer=None && skip_topk=True`：`INCOMPATIBLE`；真实路径创建了 Indexer，且当前 SFA 实现仍要求 Indexer 存在。
4. owner 写 buffer、shared 读同一 buffer：只能标 `PARTIAL/UNVERIFIED` 或更低；实际 GLM 路径未越过 layer 3 构造，不能宣称端到端 EXISTING。
5. MTP layer 保留完整 Indexer：只允许用实际 generic GLM/MTP 路径的静态证据；必须注明 MTP 未启用、运行时未验证。

同时修正：

- GLM 调用链不是 `AscendDeepseekV4ForCausalLM`；
- `GlmMoeDsaForCausalLM` 类已存在于安装的 vLLM core，不能把“config registry 是否注册”与“model class 是否存在”混为一谈；
- SFA backend 只能区分“静态平台映射已存在”和“真实 runtime forward 已验证”，当前日志未进入 attention forward；
- Stage 1 候选 allowlist 只能根据实际 generic 路径列出。`modelslim_config.py:330` 是即时报错点，不能把“吞掉缺失 quant key”当成已证明的正确修复方向；不得列入任何实际代码修改。

## 2. 重新执行合规的 construction-path eager reproduction

当前日志不合规：它显示 async scheduling、prefix caching 和 chunked prefill 均为 enabled。

1. 先保存 pinned vLLM 的 `vllm serve --help`，确认明确关闭以下特性的实际 CLI 参数：
   - async scheduling；
   - prefix caching；
   - chunked prefill。
2. 使用这些版本支持的显式关闭参数重新执行 TP=1 eager construction-path 复现。
3. 保持 graph、MTP、DSA-CP、C8、MLAPO、FlashComm/MC2、multistream、EPLB 和 P/D 关闭。
4. 新日志的 resolved config 必须证明上述特性为 disabled，而不是只在命令说明中声称。
5. 如果首错仍相同，保留完整新日志并更新 root-error 报告；如果首错变化，按时间顺序重新定位最早未恢复异常。

TP=1 结果的标题、manifest、hardware topology、eager report 和 PR body 必须统一改为：

```text
construction-path early-error reproduction only
```

禁止再称为“能容纳模型的最小运行拓扑”或模型加载/推理验证。

## 3. 补充容量拓扑证据

- 记录官方 GLM-5.2-W8A8 需要两台 Atlas 800 A2 的来源和版本。
- 给出拟用于构造缺口修复后验证的 capacity-valid 两机 TP/DP/EP 计划、每节点卡数和总 HBM。
- 明确标记 `NOT EXECUTED IN STAGE 0`。
- 本补证任务不要求在未修代码上执行两机复现，因为同一 layer 3 构造错误发生在容量验证之前。

## 4. 完成 checkpoint 冻结

补充并纳入 SHA256：

- 模型来源、repo/revision；若是本地派生权重，记录生成/复制来源和责任人；
- 完整模型文件清单、大小和 SHA256，至少包含全部 safetensors、index、config、tokenizer、generation config、chat template 和 `quant_model_description.json`；
- `config.json` digest 和关键字段摘要；
- `quant_model_description.json` digest、scheme 计数、owner/shared Indexer key 分布和 MTP key 摘要；
- 模型实际如何进入容器。当前 `docker inspect` 只记录 `/data/zyg` bind，但命令使用 `/data/czh/GLM-5.2-w8a8`，必须解释并提交可核验的 mount/copy/provenance 证据。

## 5. 修正 artifact 一致性

- `manifest.json` 中 container ID 必须与 `container-inspect.json`、create command、environment 和 devices/mounts 一致；
- `sha256sums.txt` 不得包含无法自洽的自身哈希。删除自哈希条目，或采用不会循环依赖的外部 manifest；
- 为所有新建/修订证据重新生成 SHA256；
- `manifest.json` 在复审通过前不得写 `COMPLETE`；
- `repos.txt` 应表述为“生产 tracked paths 无改动，存在 Stage 0 docs”，不能把存在 untracked docs 的工作树称为完全 clean；
- PR body 必须移除 “all five invariants are EXISTING” 和 “Stage 0 complete”，改为当前真实状态与修正后的矩阵结论。

## 6. 完成标准

- 同一 Draft PR #1 增加补证 commit，无生产源码或控制面修改；
- matrix、实际调用链、eager report、manifest、hardware topology 和 PR body 相互一致；
- 合规 TP=1 construction-path 重跑日志已提交，resolved config 显示禁用项确实关闭；
- checkpoint 文件/revision/digest 完整；
- container/model provenance 和 SHA256 清单自洽；
- 两机容量要求和未执行的 capacity-valid topology 计划已记录；
- Stage 1 candidate 仅为待审范围，不含代码实现或未经证据支持的修复结论。

完成后停止，等待 Stage 0 复审。不得开始 Stage 1。
