# Stage 0 任务：冻结证据并复现最小 eager 首错

## 任务身份

- Task ID：`GLM52-W8A8-STAGE-0`
- 执行者：DeepSeek
- 控制面维护者：技术规划与代码审查者
- 当前状态：Ready
- 生产源码修改：禁止
- 下一阶段：未授权；本任务完成后必须等待真实 Draft PR 验收

## 1. 目标

在不修改 `vllm_fl/` 生产源码的条件下：

1. 冻结实际代码、容器、软硬件和 GLM-5.2-W8A8 checkpoint；
2. 核对 shared Indexer 的五项最小不变量；
3. 在实际 NPU 上运行最小 eager 命令；
4. 把第一次真实失败定位到明确调用层、文件、类或符号；
5. 形成 Stage 1 候选文件 allowlist，但不得实现补丁。

## 2. 分支和 Draft PR

1. 执行前重新读取：

   ```bash
   git ls-remote https://github.com/xiemingda-1002/vllm-plugin-FL.git \
     refs/heads/ascend-model-migration
   ```

2. 记录返回 SHA。审计时 SHA 是 `82f3e7181cda8b51b5c0de8dd3450f5e779df363`，但禁止假设它仍未变化。
3. 同时读取本 fork 的基线：

   ```bash
   git ls-remote https://github.com/yanceng305-collab/vllm-plugin-FL.git \
     refs/heads/ascend-model-migration
   ```

4. 两个远端 SHA 必须完全一致。若本 fork 落后，只能进行可证明的 fast-forward 同步；若已分叉或无法安全对齐，停止任务并报告，不得 force-push、merge 或自行改写基线。
5. 从这个已经对齐的准确 SHA，在 `yanceng305-collab/vllm-plugin-FL` 创建：

   ```text
   audit/glm52-w8a8-stage0-gap
   ```

6. Draft PR 必须创建在 `yanceng305-collab/vllm-plugin-FL` 内，base 使用该仓库的 `ascend-model-migration`。当前不得向 `xiemingda-1002` 创建跨 fork PR。
7. Draft PR 标题建议：

   ```text
   [Audit][GLM-5.2] Freeze W8A8 Ascend gap and eager repro
   ```

8. PR 必须保持 Draft，不得合并。
9. control branch `project/glm52-w8a8-control` 不是本 PR 的 base。
10. 迁移和优化成熟后，面向同事仓库的正式 PR 将单独整理；不属于本任务。

## 3. 允许修改范围

只允许新增 Stage 0 的审计/复现资料，例如：

```text
docs/glm52-w8a8-stage0/
tests 或 tools 下独立的只读环境/manifest 收集脚本
脱敏后的 workload manifest 和结果索引
```

禁止：

- 修改 `vllm_fl/` 生产逻辑；
- 修改 `docs/glm52-w8a8-migration/` 控制面；
- cherry-pick 或复制任何上游实现；
- 修改 `.so`、`.o`、AscendC kernel 或 vendor 全栈；
- 为“试试看”开启 graph、MTP、C8 或通信优化。

如果确实需要一段收集脚本，它只能读取环境/模型 metadata 或发起复现命令，不能 monkey-patch runtime。

## 4. 子任务 0A-0：建立干净、可复现的 FL Ascend 容器

本子任务必须在后续环境冻结、shared Indexer 审计和最小 eager 复现之前完成。

### 4.1 参考范围

正式环境参考资料为控制面仓库中的 [`Qwen3.5-Ascend迁移方案与复现.md`](../Qwen3.5-Ascend迁移方案与复现.md)。Stage 0 不得再依赖服务器本地副本或历史聊天附件；只允许参考该文件 Testing 步骤 0-2，并仅复用：

- `quay.io/ascend/vllm-ascend:v0.20.2rc1` 基础镜像；
- NPU device、driver、firmware、npu-smi 和工作目录挂载方式；
- 卸载镜像自带 `vllm-ascend`；
- editable 安装 `vllm-plugin-FL`。

明确禁止从 Qwen 文档复用：

- TP=2；GLM-5.2 的拓扑必须根据实际模型容量和硬件重新确定；
- Qwen 模型路径、启动命令、prompt 或性能数字；
- graph / `FULL_DECODE_ONLY`；
- async scheduling；
- MTP/speculative decoding；
- FlashComm、multistream、shared-expert overlap；
- Qwen 的其他环境优化参数和 additional-config。

### 4.2 新建容器

1. 拉取并记录不可变镜像身份：

   ```bash
   docker pull quay.io/ascend/vllm-ascend:v0.20.2rc1
   docker image inspect quay.io/ascend/vllm-ascend:v0.20.2rc1
   ```

2. 使用独立、可识别的容器名。容器创建命令以以下结构为参考，但必须按实际服务器展开并保存最终完整命令：

   ```bash
   docker run -d \
     --name <glm52-stage0-container> \
     --net=host \
     --ipc=host \
     --privileged \
     <逐个列出实际 /dev/davinciN 的 --device 参数> \
     --device /dev/davinci_manager \
     --device /dev/devmm_svm \
     --device /dev/hisi_hdc \
     -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
     -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
     -v <主机实际 npu-smi 路径>:<容器内 npu-smi 路径> \
     -v <Stage-0 工作目录>:<容器内工作目录> \
     quay.io/ascend/vllm-ascend:v0.20.2rc1 \
     sleep infinity
   ```

   不得照搬 Qwen 的设备数量。最终命令必须逐个记录实际 device 和每个 bind mount，路径占位符不能出现在验收产物中。

3. 保存：

   - `docker pull` 输出；
   - image ID、RepoDigest 和完整 `docker image inspect`；
   - 最终 `docker run` 命令；
   - `docker inspect <container>`；
   - 实际 devices、binds、network、IPC、privileged 状态；
   - 容器名、创建时间和服务器标识。

### 4.3 核对和清理容器软件

在容器内记录实际值，不得只引用镜像文档声称的版本：

- Python；
- vLLM 版本、安装路径和源码/git 状态；
- torch；
- torch-npu；
- triton-ascend；
- transformers；
- CANN runtime/toolkit、driver、firmware；
- `npu-smi info`；
- `pip freeze` 和相关 `pip show`。

卸载镜像自带的 `vllm-ascend`，避免其 platform/general plugin 与 FL vendored Ascend backend 同时生效：

```bash
python -m pip uninstall -y vllm-ascend
python -m pip show vllm-ascend
```

第二条命令应确认 package 不存在。保存卸载前后的 package/plugin 状态和完整输出。

不得卸载镜像中的 vLLM、torch、torch-npu 或 triton-ascend 后换成未批准版本。若镜像内实际版本与预期不一致，停止并报告，不得自行升级/降级修补。

### 4.4 安装准确 SHA 的 vllm-plugin-FL

1. 使用第 2 节已经确认与上游完全对齐的 `yanceng305-collab/vllm-plugin-FL:ascend-model-migration` 准确 SHA。
2. 在干净工作目录 checkout 该 SHA；记录：

   ```bash
   git rev-parse HEAD
   git status --short --branch
   git remote -v
   ```

3. 建议 editable 安装，不解析或替换镜像中的核心依赖：

   ```bash
   python -m pip install -e <容器内 vllm-plugin-FL 路径> \
     --no-build-isolation --no-deps
   ```

4. 安装后保存：

   - 安装命令和完整日志；
   - `pip show vllm-plugin-fl`；
   - `import vllm_fl` 的模块路径；
   - FL entry point/plugin discovery 状态；
   - 安装目录的 `git rev-parse HEAD` 和干净状态；
   - `pip freeze`。

Stage 分支允许随后增加本任务规定的审计文档/只读脚本，但必须证明 `vllm_fl/` 相对上述准确基线没有变化。

### 4.5 已有容器的复用门槛

服务器上已有容器只有同时提供以下证据时才能复用：

- 基础镜像 RepoDigest 与本任务一致；
- container inspect 中 devices、mounts、network/IPC 与本任务一致；
- `vllm-ascend` 已卸载且没有残留 plugin；
- vLLM、torch、torch-npu、triton-ascend、transformers 和 CANN 实际版本一致；
- editable 安装指向准确 FL SHA，工作树无生产源码改动；
- 容器是本项目专用，能够证明没有其他实验留下的 package、patch、环境变量、kernel/cache 或源码修改影响 Stage 0。

任一条件无法证明，就不得直接复用，必须新建独立容器。清理或复用判断的证据也必须进入 artifact。

### 4.6 Bootstrap 完成门

完成后先保存容器证据和 SHA256。只有确认：

- NPU 在容器内可见；
- package 版本已冻结；
- `vllm-ascend` 已卸载；
- FL 安装 SHA 正确且生产源码干净；
- 没有启用 Qwen 的 TP=2 或任何 Stage 0 禁用优化；

才能继续子任务 0A～0D。Bootstrap 失败时停止，不得在来源不明或被其他实验修改的环境中继续审计。

## 5. 子任务 0A：冻结代码和环境

必须保存：

- 同事 fork、官方 FL、执行分支的 remote、branch、HEAD、merge-base；
- `git status --short --branch` 和完整 diff 状态；
- 安装的 vLLM package version、git SHA（若存在）及安装来源；
- vllm-plugin-FL version/SHA；
- Python、torch、torch-npu、transformers、triton-ascend、ModelSlim；
- CANN、driver、firmware、kernel package；
- 容器 image 名称和不可变 digest；
- NPU 型号、节点数、卡数、每卡 HBM、device mapping；
- NUMA、CPU、内存、NIC/HCCL interface；
- TP/DP/EP/PP 计划以及为什么这是能容纳模型的最小拓扑。

特别检查：

- 仓库 Dockerfile 是否仍默认 vLLM 0.19.0；
- 实际 runtime 是否为 vLLM 0.20.2；
- vendored A2/910B binary 是否与目标 A2/A3、CANN 版本一致；
- wheel 是否真正包含运行所需的 custom-op 和 kernel 文件。

除原有环境字段外，必须把 0A-0 的最终容器创建命令、image digest、`docker inspect`、device/mount、卸载/安装日志、容器内 package 版本和 FL 安装 SHA 纳入环境冻结结果。

## 6. 子任务 0B：冻结 checkpoint

记录：

- 模型 repo、revision、下载来源和本地路径；
- 所有权重文件的名称、大小、SHA256；
- tokenizer、`config.json`、`generation_config.json`、chat template digest；
- `quant_model_description.json` 的 SHA256、version、顶层量化类型；
- `W8A8`、`W8A8_DYNAMIC`、`FLOAT` 等 value 计数；
- routed expert、shared expert、router、dense layer、attention、Indexer、MTP 的代表 key 与量化类型；
- 哪些 layer 存在 `self_attn.indexer.*` 权重；
- 哪些 layer 是 `indexer_types=shared`；
- owner/shared pattern 是否与 `index_topk_freq`、`index_skip_topk_offset` 一致。

不得把模型名称中的 “W8A8” 当成每一层的真实 scheme。

完整量化描述若过大或包含客户限制，保存本地原件并在 PR 中提交脱敏统计、digest 和可审查 key 样本。

## 7. 子任务 0C：核对五项最小不变量

对实际安装的 vLLM 代码和同事 FL 分支做符号级检查，输出矩阵：

| 不变量 | Existing / Missing / Incompatible | repo@SHA | path:symbol | 直接证据 |
|---|---|---|---|---|
| 一个模型级共享 `topk_indices_buffer` | | | | |
| 依据 `indexer_types` 区分 owner/shared | | | | |
| shared 层 `indexer=None && skip_topk=True` | | | | |
| owner 写 buffer、shared 读同一 buffer | | | | |
| MTP layer 保留完整 Indexer | | | | |

必须继续检查以下调用链，不能只搜索文件名：

```text
GlmMoeDsaForCausalLM
  -> DeepseekV2Model
  -> DeepseekV2DecoderLayer
  -> DeepseekV2MLAAttention
  -> FL OOT MultiHeadLatentAttentionWrapper
  -> platform attention backend selection
  -> AscendSFABackend / AscendSFAImpl
  -> ModelSlim W8A8 dispatch
  -> FusedMoE
  -> KV cache spec/model runner
```

额外回答：

- Ascend 环境是否因为 early return 绕过 FL 本地 `GlmMoeDsaConfig` 注册；
- `glm_moe_dsa` 是否来自实际 vLLM core；
- backend 是否确实选择 `AscendSFABackend`；
- ModelSlim 是否能为 checkpoint 中每类 linear/MoE key 找到 scheme；
- owner/shared 层当前各自分配什么 KV/cache 结构；
- MTP patch 是否仅被导入但因未传 speculative config 而保持不执行。

## 8. 子任务 0D：最小 eager 复现

### 8.1 配置原则

必须使用：

- `VLLM_PLUGINS=fl`
- `--enforce-eager`
- `--quantization ascend`
- `--trust-remote-code`
- 最小可运行的 `max-model-len`
- `max-num-seqs=1`
- 固定 seed
- 能容纳权重的最小 TP/EP 拓扑

必须关闭或不传：

- speculative/MTP
- compilation/ACL graph
- prefix caching
- async scheduling
- DSA-CP
- sparse SFA/LI C8
- MLAPO
- FlashComm、MC2、Fused-MC2
- multistream/shared-expert overlap
- EPLB/balance scheduling
- P/D disaggregation

EP 若是容纳模型的必要条件可以使用，但必须说明它是容量条件，不是本阶段性能优化。

### 8.2 执行顺序

1. 检查 `vllm serve --help`，确认当前 CLI 参数语义。
2. 清理或记录所有会影响行为的 `VLLM_*`、`HCCL_*`、`ASCEND_*` 环境变量。
3. 执行一次冷启动，保存所有节点/rank 完整 stdout/stderr。
4. 若服务成功：
   - 发送一个固定短 prompt，temperature=0；
   - 保存文本、token IDs、finish reason、首 token 和 HBM；
   - 再执行至少一次重启，确认不是残留 cache 偶然成功。
5. 若服务失败：
   - 不修改代码；
   - 保存首个根错误及其前后上下文；
   - 将后续 worker shutdown、RPC timeout 等标成下游噪声；
   - 定位到上述调用链中的具体层。

不要通过同时改变版本、拓扑、代码和多个开关来“试到能跑”。每次诊断性重试只能改变一个已记录变量。

## 9. 必须保存的产物

建议 artifact 结构：

```text
stage-0/
  manifest.json
  repos.txt
  environment.txt
  pip-freeze.txt
  container/
    create-command.sh
    image-inspect.json
    container-inspect.json
    devices-and-mounts.txt
    package-versions.txt
    uninstall-vllm-ascend.log
    install-vllm-plugin-fl.log
  npu-smi.txt
  hardware-topology.txt
  model-manifest.json
  model-files.sha256
  config-summary.json
  quant-summary.json
  symbol-gap-matrix.md
  commands/
  logs/node-*/
  outputs/
  sha256sums.txt
```

完整日志和大文件必须保存，并包含在 `sha256sums.txt` 中。存储规则：

- 如果已经有现成 artifact 存储，可以上传并记录 URL；
- 如果没有，允许保存在服务器固定目录；
- 使用服务器目录时，PR 中必须提交 artifact 索引、固定绝对路径、SHA256、保留责任/期限说明和必要的脱敏日志片段；
- artifact URL 不是 Stage 0 完成的硬性条件；
- 不得为了上传 Stage 0 日志额外搭建 GitHub Actions、workflow 或其他 artifact 基础设施。

PR body 必须包含：

- base/head SHA；
- 容器创建或复用证明、image digest、device/mount、容器内 package 版本、`vllm-ascend` 卸载状态和 FL 安装 SHA；
- 完整命令；
- artifact 存储定位信息和 SHA256：现成存储 URL，或服务器固定目录与 artifact 索引；
- 第一次失败或成功点；
- 已确认事实、推断和未知项；
- Stage 1 候选文件 allowlist；
- 已知问题和未执行项目。

不得提交 token、密码、私钥、cookie、内部凭据、客户 prompt 原文或未经允许的敏感权重路径。

## 10. 完成标准

Stage 0 只有同时满足以下条件才算完成：

- Draft PR 已创建且保持 Draft；
- Draft PR 位于 `yanceng305-collab/vllm-plugin-FL`，base 为本 fork 的 `ascend-model-migration`，没有向 `xiemingda-1002` 创建跨 fork PR；
- 本 fork `ascend-model-migration` 与执行时读取的上游 SHA 完全一致，Stage 分支基于这个准确、可验证的 SHA；
- 0A-0 bootstrap 已通过：独立容器基于准确 image digest，实际 NPU/device/mount 可核验，`vllm-ascend` 已卸载，FL editable 安装 SHA 正确；或已有专用容器满足并证明全部等价条件；
- 容器创建命令、image/container inspect、device/mount、package 版本和卸载/安装日志已进入 artifact 并生成 SHA256；
- 生产源码没有改动；
- 环境和 checkpoint 身份可复现；
- 五项 shared Indexer 不变量矩阵完整；
- 最小 eager 命令和所有节点日志完整；
- 完整日志和大文件已有 SHA256；提供现成 artifact URL，或服务器固定目录、artifact 索引和必要的脱敏片段；
- 首次失败定位到具体符号，或未经修改成功的证据可复验；
- 提供由证据支持的 Stage 1 候选 allowlist；
- 未提前启用 Stage 1～4 特性。

完成后停止工作，等待控制面维护者审查真实 Draft PR、diff、日志和环境证据。不得自行开始 Stage 1。
