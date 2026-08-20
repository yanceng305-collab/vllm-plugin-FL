#!/bin/bash
# Stage 0 Container Creation Command
# Generated: 2026-08-20T03:07:17Z

docker pull quay.io/ascend/vllm-ascend:v0.20.2rc1

docker run -d --name fl_stage0_audit \
  --net=host --ipc=host --privileged \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /data/zyg:/data/zyg \
  quay.io/ascend/vllm-ascend:v0.20.2rc1 \
  sleep infinity

# Container ID: bef74a51a808eb161152b202bfb44d273097ac88015d60bd2844e061b1efa82d
# Image RepoDigest: quay.io/ascend/vllm-ascend@sha256:a926b560b81785a9f820f975748a39807f0b1753675ab40ec30cb4ae3dabb71e