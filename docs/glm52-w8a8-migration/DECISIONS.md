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
