# GLM-5.2-W8A8 迁移状态

- 最后更新：2026-08-20（Asia/Shanghai）
- 当前阶段：Stage 0 — 证据冻结与 eager 首错复现
- 当前状态：任务已发布，等待 DeepSeek 开始；尚未验收
- 执行分支：未创建；先确认 `yanceng305-collab/vllm-plugin-FL:ascend-model-migration` 与执行时读取的上游 `xiemingda-1002` HEAD 完全一致，再从该准确 SHA 在本 fork 创建 `audit/glm52-w8a8-stage0-gap`
- Draft PR：未创建；创建后 base 必须是 `yanceng305-collab/vllm-plugin-FL:ascend-model-migration`，当前不得向 `xiemingda-1002` 创建跨 fork PR
- 代码基线：上游和本 fork 审计时均为 `82f3e7181cda8b51b5c0de8dd3450f5e779df363`；执行前必须重新读取并比较两个远端 SHA，禁止只使用浮动分支名
- 执行 HEAD：N/A
- 控制面基线：`main@38e7dbc20197e2db742c4e4c9687d36ea4df9900`
- 最近验收结果：总体计划和上游/分支缺口审计已完成；Stage 0 尚无执行证据，因此没有通过结论
- 当前已知约束：实际 NPU 拓扑、runtime 版本、checkpoint digest 和客户 benchmark 附件尚未由执行环境冻结
- 下一动作：将 [`tasks/STAGE-0.md`](tasks/STAGE-0.md) 直接交给 DeepSeek；先同步/确认本 fork 的代码基线，再在本 fork 创建 Stage 0 Draft PR。完整日志可使用现成 artifact URL，或保存在服务器固定目录并提交索引、路径和 SHA256
