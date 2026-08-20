# GLM-5.2-W8A8 迁移状态

- 最后更新：2026-08-20（Asia/Shanghai）
- 当前阶段：Stage 0 — 证据冻结与 eager 首错复现
- 当前状态：`NEEDS-FIX`；Stage 0 Draft PR #1 已正式审查，Stage 1 仍未授权
- 执行分支：`audit/glm52-w8a8-stage0-gap`
- Draft PR：[yanceng305-collab/vllm-plugin-FL#1](https://github.com/yanceng305-collab/vllm-plugin-FL/pull/1)，状态 OPEN / DRAFT，base 为本 fork `ascend-model-migration`
- 代码基线：上游和本 fork 当前均为 `82f3e7181cda8b51b5c0de8dd3450f5e779df363`
- 执行 HEAD：`637a54978eff7bf7fae0c6150fe7ca663b22461a`
- 控制面基线：`main@38e7dbc20197e2db742c4e4c9687d36ea4df9900`
- 最近验收结果：真实 diff 仅含 17 个 Stage 0 证据文件，未修改生产源码；容器 bootstrap 基本成立；首个根错误高置信确认为 layer 3 shared 层错误创建 Indexer 后触发 `KeyError: model.layers.3.self_attn.indexer.wq_b.weight`。但 symbol matrix 使用未执行的 DeepSeek-V4 路径，把第 2/3/4 项错误标为 EXISTING；实际 eager 配置仍启用 async scheduling、prefix caching 和 chunked prefill；checkpoint 文件/revision/digest 未冻结；artifact manifest 还存在容器 ID 和自哈希不一致。因此未通过。
- 拓扑结论：TP=1、单机 8×64GB A2 证据只允许标为 construction-path early-error reproduction，不是可容纳 GLM-5.2-W8A8 的最小运行拓扑。Stage 0 不要求在修复构造缺口前重复同一错误的两机运行，但必须补充官方两机 A2 容量要求和未执行的 capacity-valid topology 计划；构造缺口修复后的首次 eager 验证必须使用容量有效的两机拓扑。
- 下一动作：DeepSeek 只在同一 Draft PR #1 中执行 [`tasks/STAGE-0-FIX-1.md`](tasks/STAGE-0-FIX-1.md)，补正 Stage 0 证据和 PR 描述；不得修改生产源码、不得开始 Stage 1、不得合并
