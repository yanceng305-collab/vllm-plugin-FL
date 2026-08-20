# GLM-5.2-W8A8 Ascend 迁移长期计划

## 1. 文档定位

本目录是 GLM-5.2-W8A8 迁移到 Ascend NPU、通过 `vLLM + vllm-plugin-FL` 运行并继续优化的长期控制面。

- GitHub control branch 是项目规划、状态和技术决策的唯一事实源。
- 聊天记录、个人笔记、commit message 中未附证据的描述均不能替代本目录。
- 控制面由技术规划与代码审查者维护。
- DeepSeek 只负责实现、实验和提交证据，不得直接修改本目录。
- 每阶段通过真实 Draft PR、diff、环境、日志和 benchmark 验收后，才生成下一阶段详细任务。

当前可执行任务只见 [`tasks/STAGE-0.md`](tasks/STAGE-0.md)。Stage 1～4 暂不提供详细执行文件。

## 2. 项目目标

1. 在固定版本的 vLLM、vllm-plugin-FL、Ascend 软件栈和 GLM-5.2-W8A8 checkpoint 上完成最小 eager 推理。
2. 验证权重加载、shared Indexer/IndexCache、SFA、MoE、KV Cache 和输出正确性。
3. 建立可复现的 W8A8/NPU eager baseline。
4. 根据 profiler 逐项优化；每次只改变一个可归因变量。
5. 保持 GPU FP8/H100 与 Ascend ModelSlim W8A8 结果相互隔离。

## 3. 当前证据基线

### 3.1 仓库与版本

- FL 官方 main：`38e7dbc20197e2db742c4e4c9687d36ea4df9900`
- 同事迁移分支：`xiemingda-1002/vllm-plugin-FL:ascend-model-migration`
- 审计时同事分支 HEAD：`82f3e7181cda8b51b5c0de8dd3450f5e779df363`
- 同事分支的 Ascend 基础来自大规模 DeepSeek-V4 vendor 导入；唯一直接命名 GLM 的独有提交 `f502d616c953b4c0383986db95fc9dfc4a282e12` 只处理 graph 捕获前的 custom-op 加载。
- 同事分支测试依赖指向 vLLM 0.20.2，但 Docker 默认版本、runtime dependency 和实际环境存在歧义，必须在 Stage 0 重新冻结。

固定证据：

- [同事分支 HEAD](https://github.com/xiemingda-1002/vllm-plugin-FL/commit/82f3e7181cda8b51b5c0de8dd3450f5e779df363)
- [DeepSeek-V4 Ascend 导入](https://github.com/xiemingda-1002/vllm-plugin-FL/commit/097bd298f78202c3fa5ac7be56cfdf93be1774e1)
- [GLM graph-only 补丁](https://github.com/xiemingda-1002/vllm-plugin-FL/commit/f502d616c953b4c0383986db95fc9dfc4a282e12)
- [FL 官方 main](https://github.com/flagos-ai/vllm-plugin-FL/commit/38e7dbc20197e2db742c4e4c9687d36ea4df9900)

### 3.2 已确认的技术缺口

GLM-5.2 并不需要复制一套新的大模型实现。vLLM 0.20.2 已有 `GlmMoeDsaForCausalLM` 薄别名和基础 DSA 模型，但缺少 GLM-5.2 checkpoint 所需的 shared Indexer 语义：

1. 模型级共享 `topk_indices_buffer`；
2. 依据 `indexer_types` 区分 owner/shared 层；
3. shared 层不创建或加载本地 Indexer；
4. owner 层写 top-k，shared 层复用同一 buffer；
5. Ascend SFA 支持 `indexer=None && skip_topk=True`。

必须区分两个概念：

- `skip_topk` 表示当前层不重新计算 top-k；
- 只有 `indexer_types[layer] == "shared"` 才证明 checkpoint 没有该层 Indexer 权重。

主要上游参考：

- [vLLM #44420：IndexCache、pattern、offset、MTP buffer plumbing](https://github.com/vllm-project/vllm/pull/44420)
- [vLLM #45895：shared 层跳过 Indexer 初始化](https://github.com/vllm-project/vllm/pull/45895)
- [vllm-ascend #10441：GLM-5.2 shared-layer/SFA 适配](https://github.com/vllm-project/vllm-ascend/pull/10441)
- [vllm-ascend #11363：修正 checkpoint 所有权判断](https://github.com/vllm-project/vllm-ascend/pull/11363)

这些 PR 只能作为行为与符号级参考，禁止整块复制到 vLLM 0.20.2/FL。

### 3.3 实际 Ascend attention 路径

GLM-5.2 的 `glm_moe_dsa` 在已核验的 Ascend 0.23 路径中使用：

```text
GlmMoeDsaForCausalLM
  -> DeepseekV2Model / DeepseekV2MLAAttention
  -> MultiHeadLatentAttentionWrapper
  -> AscendSFABackend / AscendSFAImpl
  -> lightning indexer or cached top-k
  -> sparse flash attention
```

它不是 DeepSeek-V4 使用的 compressed `AscendDSABackend`。当前 vLLM 的 `deepseek_v32/nvidia` 还是 CUDA 条件选择路径，不得移植到 Ascend。

### 3.4 W8A8 checkpoint 语义

公开 `Eco-Tech/GLM-5.2-w8a8` 的 ModelSlim 描述显示：

- routed/shared expert 矩阵主要为 `W8A8_DYNAMIC`；
- router 为 `FLOAT`；
- MTP expert 为 `FLOAT`；
- Indexer 权重只存在于 owner 层和 MTP 层。

现有同事分支已经包含 `glm_moe_dsa` packed mapping 和 W8A8_DYNAMIC linear/MoE 路径。是否需要修改量化代码，只能由项目实际 checkpoint manifest 和真实加载错误决定。

- [GLM-5.2 config](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json)
- [公共 W8A8 quant_model_description](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8/resolve/master/quant_model_description.json)

## 4. 控制面和代码分支治理

### 4.1 控制面

- 长期分支：`project/glm52-w8a8-control`
- 控制面只保存计划、状态、决策和经批准的阶段任务。
- 控制面更新由规划/审查者提交。
- DeepSeek 不得在任何执行 PR 中修改 `docs/glm52-w8a8-migration/`。

### 4.2 代码与实验

- Stage 0 及后续开发 Draft PR 暂时只在 `yanceng305-collab/vllm-plugin-FL` 内进行。
- 每个 Stage 开始前仍须读取 `xiemingda-1002/vllm-plugin-FL:ascend-model-migration` 的真实 HEAD；只有确认本 fork 的 `ascend-model-migration` 与该 SHA 完全一致后，才能从这个准确 SHA 创建 Stage 分支。
- Stage 0 Draft PR 的 base 是本 fork 的 `ascend-model-migration`。当前不得向 `xiemingda-1002` 创建跨 fork PR。
- 后续代码阶段继续使用 stacked Draft PR：每阶段从上一阶段已验收 HEAD 创建，PR base 指向本 fork 中上一阶段代码分支。
- GLM-5.2 迁移和优化成熟后，再单独整理面向同事仓库的正式 PR；该正式 PR 不属于当前 Stage 0 流程。
- control branch 永远不作为代码 PR base。
- 不允许自动合并、force-push 已验收分支或把多个阶段 squash 成不可审计的大提交。

### 4.3 阶段验收循环

```text
控制面发布阶段任务
  -> 执行者创建独立代码/实验分支和 Draft PR
  -> 执行者提交 diff、环境、日志、结果和已知问题
  -> 审查者核验真实 GitHub PR 和原始证据
  -> 审查者更新 STATUS.md / DECISIONS.md
  -> 通过后才创建下一阶段任务
```

## 5. 阶段计划

### Stage 0：证据冻结和 eager 首错复现

状态：当前唯一已授权执行阶段。

目标：不修改生产源码，先建立或证明可复用一个基于 `quay.io/ascend/vllm-ascend:v0.20.2rc1` 的独立、干净、可复现 FL Ascend 容器，再冻结版本、硬件、checkpoint 和运行命令，核对 shared Indexer 五项不变量，并将第一次真实失败定位到具体调用层。

容器 bootstrap 的正式环境参考是本控制面内的 [`Qwen3.5-Ascend迁移方案与复现.md`](Qwen3.5-Ascend迁移方案与复现.md)，且只允许使用其中 Testing 步骤 0-2 的镜像、device/mount、卸载 `vllm-ascend` 和 editable 安装流程。不得再依赖服务器本地副本或历史聊天附件。Qwen 的 TP=2、模型命令及 graph、async scheduling、MTP、FlashComm、multistream、prefix caching 等参数不适用于 GLM-5.2 Stage 0。

详细任务：[`tasks/STAGE-0.md`](tasks/STAGE-0.md)

Stage 0 验收后才能决定 Stage 1 的准确文件 allowlist。

### Stage 1：最小 eager 兼容

尚未授权，暂不生成详细任务。

预期门槛：只解决由 Stage 0 证据确认的模型构造、shared Indexer/SFA、ModelSlim 或阻塞性 KV 问题；不得同时加入 graph、MTP、DSA-CP、C8 或通信优化。

### Stage 2：W8A8/NPU eager baseline

尚未授权，暂不生成详细任务。

预期门槛：在已验收 eager commit 上冻结 correctness、稳定性和性能 workload，保存 TTFT/TPOT/ITL/E2E、吞吐、HBM、KV blocks 和长时运行结果。

### Stage 3：profiling 和单一优化选择

尚未授权，暂不生成详细任务。

预期门槛：分离 prefill/decode，定位 Indexer/SFA、MoE/HCCL、host launch 或 KV/HBM 瓶颈，只批准一个下一变量。

### Stage 4：单变量优化循环

尚未授权，暂不生成详细任务。

候选包括 shared-layer KV 内存、SFA/Indexer、MoE/EP、ACL graph、MTP/IndexCache。每个候选必须是独立 A/B Draft PR，收益超过 baseline 噪声且没有 correctness/stability 回归；组合优化必须晚于各单项验收。

## 6. 统一证据合同

每个阶段必须提供：

| 类别 | 必需内容 |
|---|---|
| 代码身份 | repo、branch、base/head SHA、完整 diff、未提交状态 |
| 运行环境 | 容器创建命令、image digest、container inspect/device/mount、OS、Python、vLLM、FL、torch、torch-npu、CANN、driver、firmware、transformers、triton-ascend、`vllm-ascend` 卸载证明、FL 安装 SHA/状态 |
| 硬件 | NPU 型号、节点/卡数、HBM、设备映射、NUMA、NIC、TP/DP/EP/PP |
| 模型 | repo/revision、文件 SHA256、config/quant description digest、量化类型统计 |
| 命令 | server/offline/client/benchmark 完整命令和环境变量 |
| 正确性 | 固定 prompt、token IDs/hash、参考实现比较、NaN/Inf/乱码/异常重复检查 |
| 稳定性 | 冷启动、连续请求、失败数、进程重启、HBM/RSS 变化 |
| 性能阶段 | workload manifest、raw JSON、重复次数、TTFT/TPOT/ITL/E2E、吞吐、profiler |
| 原始产物 | 每节点日志、rank 日志、npu-smi、trace、SHA256，以及存储定位信息：已有 artifact 存储时记录 URL，否则记录服务器固定目录、artifact 索引和必要的脱敏片段 |

PR 中只有截图、汇总表、commit body 自述或手工复制的数字，不构成验收证据。

Stage 0 不要求为了日志上传新建 GitHub Actions 或其他 artifact 基础设施。完整日志和大文件必须保存并生成 SHA256，但 artifact URL 不是硬性完成条件。

## 7. Benchmark 隔离规则

客户 GPU Case 可以用于统一 workload，但结果命名至少包含：

```text
case_id / model_revision / weight_format / hardware / runtime_commits / mode / run_id
```

- GPU：`GLM-5.2-FP8/H100` 或其他实际格式/硬件；
- NPU：`GLM-5.2-ModelSlim-W8A8/<实际 Ascend>`。

GPU 性能、Ascend W4A8C8 结果、Qwen 迁移结果均不得成为 W8A8/NPU 通过阈值。现有 FL “mixed-length”脚本只随机执行多个固定长度桶，不是真实 mixed-request traffic，报告中必须正确命名。

## 8. 控制面维护规则

- 当前执行事实只写入 `STATUS.md`。
- 稳定技术判断、被否决路线及其证据写入 `DECISIONS.md`。
- `PLAN.md` 只在总体目标、阶段边界或治理方式发生变化时修改。
- 新阶段任务只有在上一阶段验收后才能创建。
- 执行者提出控制面修改建议时，应在 Draft PR 评论或验收材料中说明，由审查者决定并提交。
