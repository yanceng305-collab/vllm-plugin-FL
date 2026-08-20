# GLM-5.2-W8A8 技术决策记录

本文件只记录会影响后续执行路线的重要判断。每项决策必须区分事实、推断和适用边界。

## D-001：GitHub control branch 是唯一事实源

- 状态：Accepted
- 日期：2026-08-20
- 决策：项目状态、阶段任务和技术判断以 `project/glm52-w8a8-control` 中本目录为准；聊天上下文不作为长期状态。
- 原因：项目会跨多个模型会话和执行者，聊天可能被压缩或丢失。
- 边界：代码实现的真实状态仍以对应 GitHub Draft PR、commit、日志/大文件及其 SHA256 和存储定位信息为原始证据；控制面只记录验收结论和链接。

## D-002：控制面发布到独立个人 fork，不写入同事代码分支

- 状态：Superseded by D-010
- 日期：2026-08-20
- 决策：控制面位于 `yanceng305-collab/vllm-plugin-FL:project/glm52-w8a8-control`，从 `main@38e7dbc` 建立。
- 证据：当前 GitHub 身份对 `xiemingda-1002/vllm-plugin-FL` 的权限为 read/pull，没有 push 权限。
- 原因：保持规划和 `ascend-model-migration` 代码历史分离，同时让 DeepSeek 和后续审查会话可以公开读取。
- 边界：代码 Draft PR 仍从同事 `ascend-model-migration` 的准确 HEAD 创建并采用 stacked PR；不得以 control branch 为 base。

## D-003：第一缺口是 shared Indexer/SFA 语义，不是整套 GLM 模型

- 状态：Accepted
- 日期：2026-08-20
- 决策：不得复制当前 vLLM 的完整 GLM/DeepSeek-V3.2/NVIDIA model subtree。优先核对并最小适配 model-level top-k buffer、owner/shared 层、conditional Indexer 和 Ascend SFA。
- 证据：vLLM 0.20.2 已注册 `GlmMoeDsaForCausalLM`；0.20.2 缺少 layer-wise IndexCache/skip，GLM-5.2 config/checkpoint 又明确省略 shared 层 Indexer 权重。
- 参考：[vLLM #44420](https://github.com/vllm-project/vllm/pull/44420)、[vLLM #45895](https://github.com/vllm-project/vllm/pull/45895)、[vllm-ascend #10441](https://github.com/vllm-project/vllm-ascend/pull/10441)。

## D-004：`skip_topk` 与 checkpoint 模块所有权必须分开

- 状态：Accepted
- 日期：2026-08-20
- 决策：`skip_topk` 只控制是否重新计算；只有 `indexer_types[layer] == "shared"` 才允许省略 Indexer 模块/权重。MTP layer 必须保留完整 Indexer。
- 证据：[vllm-ascend #11363](https://github.com/vllm-project/vllm-ascend/pull/11363)。
- 风险：错误地按 `skip_topk` 删除所有 Indexer 会破坏 GLM-5/5.1 runtime IndexCache 场景。

## D-005：GLM-5.2 Ascend 第一阶段走 SFA，不走 DeepSeek-V4 compressed DSA

- 状态：Accepted
- 日期：2026-08-20
- 决策：Stage 0 必须核验实际 backend 是 `AscendSFABackend`。误走 `AscendDSABackend` 应作为接口/版本错误处理，不能通过复制 DeepSeek-V4 model 修补。
- 证据：vllm-ascend 0.23 对 `(use_mla=True, use_sparse=True, use_compress=False)` 的平台选择和 `sfa_v1.py` 路径。

## D-006：Stage 0 只收集证据，不写迁移代码

- 状态：Accepted
- 日期：2026-08-20
- 决策：先冻结 runtime、硬件、checkpoint、最小命令和首错，随后才批准 Stage 1 文件范围。
- 原因：当前分支存在 vLLM 0.19/0.20.2、CANN、A2/A3 及 checkpoint 格式歧义；未经复现直接编码会把版本问题误判为模型缺口。

## D-007：KV 内存、MTP、graph 和通信优化全部后置

- 状态：Accepted
- 日期：2026-08-20
- 决策：
  - [vllm-ascend #11065](https://github.com/vllm-project/vllm-ascend/pull/11065) 的异构 KV spec 默认在 eager 跑通后单独测试，只有真实 OOM 才允许前移；
  - [vllm-ascend #10510](https://github.com/vllm-project/vllm-ascend/pull/10510) 和 proposer/MTP 路径不进入第一阶段；
  - ACL graph、DSA-CP、C8、MLAPO、FlashComm/MC2、multistream、async、prefix cache 均在 baseline 后单变量验证。

## D-008：GPU FP8/H100 只复用 workload，不提供 NPU 目标

- 状态：Accepted
- 日期：2026-08-20
- 决策：GPU 与 NPU 结果使用不同 namespace，GPU 数值不参与 W8A8/NPU pass/fail。
- 原因：权重格式、kernel、内存层次、通信和硬件不同，直接设置性能目标没有证据基础。

## D-009：执行者不得修改控制面

- 状态：Accepted
- 日期：2026-08-20
- 决策：DeepSeek 不得修改 `docs/glm52-w8a8-migration/`。若发现计划错误，应在执行 Draft PR 评论或证据报告中提出，由控制面维护者审查后更新。

## D-010：开发 Draft PR 暂时只在本 fork 内进行

- 状态：Accepted
- 日期：2026-08-20
- 触发阶段：Stage 0 及后续开发阶段
- 决策：Stage 0 和后续开发 Draft PR 暂时只在 `yanceng305-collab/vllm-plugin-FL` 内创建。每阶段开始前读取 `xiemingda-1002/vllm-plugin-FL:ascend-model-migration` 的真实 HEAD，确认本 fork 同名分支与其完全一致，再从该准确 SHA 建立 Stage 分支。Stage 0 PR base 使用本 fork 的 `ascend-model-migration`；后续继续使用本 fork 内 stacked Draft PR。
- 当前限制：不得向 `xiemingda-1002` 创建跨 fork 开发 PR。
- 后续出口：迁移和优化成熟后，另行整理面向同事仓库的正式 PR。
- 不变边界：`project/glm52-w8a8-control` 永远不能作为代码 PR base。

## D-011：artifact URL 不是 Stage 0 硬性条件

- 状态：Accepted
- 日期：2026-08-20
- 触发阶段：Stage 0
- 决策：完整日志和大文件必须保存并生成 SHA256。已有现成 artifact 存储时可以记录 URL；没有时允许保存在服务器固定目录，并在 PR 中提交 artifact 索引、服务器路径、SHA256 和必要的脱敏日志片段。
- 禁止：不得仅为上传 Stage 0 日志额外搭建 GitHub Actions 或其他 artifact 基础设施。
- 验收边界：审查者必须能够通过索引、固定路径和 SHA256确认产物身份；只有汇总或截图仍不构成证据。

## D-012：Stage 0 先建立干净的 FL Ascend 容器

- 状态：Accepted
- 日期：2026-08-20
- 触发阶段：Stage 0
- 决策：在环境冻结、shared Indexer 审计和 eager 首错复现之前，先基于 `quay.io/ascend/vllm-ascend:v0.20.2rc1` 建立独立容器，挂载实际 NPU、driver、firmware、npu-smi 和工作目录；核对容器内 package；卸载镜像自带 `vllm-ascend`；从执行时已对齐的 `ascend-model-migration` 准确 SHA editable 安装 `vllm-plugin-FL`。
- 正式参考：本控制面内的 [`Qwen3.5-Ascend迁移方案与复现.md`](Qwen3.5-Ascend迁移方案与复现.md)。Stage 0 不再依赖服务器本地文档或历史聊天附件。
- 参考边界：只允许使用该文档 Testing 步骤 0-2 的容器/软件搭建流程。不得照搬 Qwen TP=2、模型启动命令或 graph、async scheduling、MTP、FlashComm、multistream、prefix caching 等优化参数。
- 允许复用：已有专用容器只有在证明 image digest、device/mount、`vllm-ascend` 卸载状态、实际 package 版本、FL 安装路径和代码 SHA全部一致且环境无其他实验修改后才可复用；否则必须新建。
- 证据要求：保存容器创建命令、image digest、`docker inspect`、device/mount、卸载/安装日志、容器内 package 版本和 FL git 状态，并纳入 Stage 0 artifact/SHA256。
- 不变边界：该 bootstrap 不改变 Stage 0 的模型技术目标、五项 shared Indexer 不变量、eager 禁用项或停止等待验收要求。

## D-013：Stage 0 Draft PR #1 验收结论为 NEEDS-FIX

- 状态：Accepted
- 日期：2026-08-20
- 触发阶段：Stage 0 / [Draft PR #1](https://github.com/yanceng305-collab/vllm-plugin-FL/pull/1)
- 结论：PR #1 未达到 Stage 0 完成标准，只允许补充/修正 Stage 0 证据；Stage 1 仍未授权。
- 已通过部分：base/head 和 Draft PR 关系正确；17 个变更文件均位于 `docs/glm52-w8a8-stage0/`；未修改生产源码；image digest、device/mount、`vllm-ascend` 卸载和 FL SHA 安装证据基本齐全；首个根错误定位正确。
- 根错误：eager 日志在 vLLM core `deepseek_v2.py:985` 为 layer 3 创建 Indexer，随后 ModelSlim `modelslim_config.py:330` 查询 checkpoint 中不存在的 `model.layers.3.self_attn.indexer.wq_b.weight` 并抛出 `KeyError`。错误发生在模型构造期间，后续 EngineCore/API server 异常是下游噪声。
- Matrix 修正：第 1 项可保留 EXISTING，但证据必须改为实际 vLLM core `DeepseekV2Model`；第 2 项应为 MISSING/INCOMPATIBLE；第 3 项应为 INCOMPATIBLE；第 4 项至少为 PARTIAL/UNVERIFIED，不能声称实际 GLM 路径已完成 owner-write/shared-read；第 5 项只能按实际 generic GLM/MTP 路径给出静态证据并标明运行时未验证。DeepSeek-V4 vendored model 不能替代实际 GLM 调用链证据。
- 额外缺口：日志明确显示 async scheduling、prefix caching 和 chunked prefill 为 enabled；checkpoint repo/revision、完整文件清单和 SHA256、config/quant description digest 未提交；`manifest.json` 容器 ID 与 `container-inspect.json` 不一致；`sha256sums.txt` 的自哈希不匹配；模型路径 `/data/czh/GLM-5.2-w8a8` 不在已记录 bind mount 中，来源/进入容器方式未证明。
- 修正边界：只修正文档、manifest、hash 和补充复现日志；禁止生产代码改动、Stage 1 实现或扩大 PR 范围。

## D-014：TP=1 证据降级为 construction-path reproduction

- 状态：Accepted
- 日期：2026-08-20
- 触发阶段：Stage 0 / Draft PR #1
- 决策：单台 8×64GB A2、TP=1 的当前日志只能证明模型构造路径和首个结构错误，不能称为“能容纳模型的最小运行拓扑”，也不能证明权重加载、KV 分配或推理能力。
- 两机判断：Stage 0 不要求在构造缺口尚未修复时补跑会在同一 layer 3 提前失败的两机实验；这种运行无法验证容量。Stage 0 必须记录官方 GLM-5.2-W8A8 需要两台 Atlas 800 A2、给出 capacity-valid TP/DP/EP 计划并明确未执行。
- 后续硬门：构造缺口修复后，任何“eager 跑通”、权重加载或稳定性验收必须先在容量有效的两机 A2 拓扑上完成；TP=1 结果不得升级为该结论。

## 后续决策模板

```markdown
## D-NNN：标题

- 状态：Proposed / Accepted / Rejected / Superseded
- 日期：YYYY-MM-DD
- 触发阶段：Stage N
- 决策：
- 直接证据：repo/PR/commit/path/symbol/log/benchmark/artifact
- 推断：
- 被否决方案及原因：
- 适用环境和版本：
- 后续影响：
```
