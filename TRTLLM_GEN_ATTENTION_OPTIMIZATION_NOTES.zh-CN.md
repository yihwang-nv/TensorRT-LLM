<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# trtllm-gen Attention 优化说明

本文总结本次 session 中对 trtllm-gen attention 路径做的重构、优化，以及通过
NVTX/nsys 收集到的性能数据。目标是向 maintainer 解释：当前 PR 解决了什么
问题，已经取得了哪些收益，剩余性能差距主要在哪里。

当前分支基于 #12525。这个依赖是必要的，因为 #12525 修改了
shared-paged-index 行为，并统一了 trtllm-gen FlashInfer 路径和
`thop.attention` 的 KV-cache buffer 计算。

## 背景和问题

原始 trtllm-gen attention 集成主要有三个问题：

1. Workspace layout 逻辑在 `AttentionOp` 和 `trtllm_gen.py` 中重复实现。
   `enqueueContext` / `enqueueGeneration` 里仍然用大量 `nextWorkspacePtr`
   手动拆分 workspace。
2. trtllm-gen Python 路径会在每层、每步重复构造或 materialize metadata。
   其中 KV-cache metadata 对于同一层在 KV pool 和 block table 确定后是稳定的。
3. GPT-OSS W4 1-GPU 测试中，trtllm-gen 明显慢于 `thop.attention`。两条路径
   的 trtllm-gen FMHA kernel 来自同一个 kernel generator，因此主要怀疑点是
   host-side setup 和 workspace 管理，而不是 FMHA kernel 本身。

## 已实现的优化

### 统一 Attention Workspace Manager

新增 `tensorrt_llm::common::op::AttentionWorkspaceManager`，位于
`cpp/tensorrt_llm/common/attentionWorkspace.h`。

它现在是以下 workspace layout 的单一来源：

- `AttentionOp` context workspace。
- `AttentionOp` generation workspace。
- FlashMLA workspace。
- XQA / trtllm-gen generation workspace。

Workspace manager 返回 `WorkspaceSlice {offset, size}`，并可以从 raw workspace
pointer materialize typed view struct。`AttentionOp` 改为使用这个 manager，而不是
在 `enqueueContext` / `enqueueGeneration` 中手写 workspace split。trtllm-gen torch
extension 也复用同一套 layout 顺序，并向 Python 暴露 trtllm-gen 需要的 subset。

这个改动的核心价值是：`thop.attention` 和 trtllm-gen 共享同一个 workspace
contract，避免两套 layout 漂移。

### 重构 trtllm_gen.py

`trtllm_gen.py` 被重构为面向对象的 backend 结构：

- `FlashInferTrtllmGenAttention` 负责 support check、workspace layout、
  KV-cache metadata cache、context/generation dispatch。
- `TrtllmAttentionWrapper` 只在 `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1` 时创建
  trtllm-gen backend。
- backend 按 `(kv_cache_manager, quant_config)` 对象身份缓存。
- 删除旧的 unfused trtllm-gen fallback 路径和不再使用的 torch op fake。
- 删除 Python 版本 WorkspaceManager fallback，让 nanobind/C++ workspace manager
  成为唯一 layout 来源。

这样之后，trtllm-gen fused path 是一个清晰的 backend 对象，而不是散落在文件中的
一组 free function，后续扩展更容易。

### 新增 Fused Preprocess/Postprocess Ops

新增三个内部 fused torch-extension 入口：

- `trtllm_gen_context_preprocess`
- `trtllm_gen_context_postprocess`
- `trtllm_gen_generation_preprocess`

它们替代原先每步执行的链式调用：

- `torch.ops.trtllm.build_decoder_info`
- `qkv_preprocessing`
- `kv_cache_postprocessing`

fused ops 新增 `need_build_kv_cache_metadata` flag。binding 默认值保持向后兼容，
但 Python backend 在缓存了每层的 `kv_cache`、`block_tables` 和 pool index 后，
会传 `need_build_kv_cache_metadata=False`。这样可以避免重复构造 FlashInfer
paged-KV metadata，也避免原先 C++ 中读取 pool mapping 造成的 host sync 点。

### C++ Hot Path 简化

fused preprocess 代码现在按内部 C++ 热路径处理，而不是按公开 torch op wrapper
处理。主要减少了以下开销：

- 直接 inline 构造 `BuildDecoderInfoParams`。
- 删除仅转发到单一内部调用点的 wrapper，例如 `build_decoder_info`、
  `invokeBuildDecoderInfoTyped`、`qkv_processing<true/false>`。
- 使用 `WorkspaceAccessor` 缓存 workspace base pointer、tensor options 和
  element-size 计算。
- 删除 `makeOptionalWorkspaceTensorView`。现在只创建 FlashInfer 或 fused kernel
  确实需要的 tensor view。
- CUDA stream 不做跨调用缓存，因为 current stream 是 PyTorch 状态，可能在调用间变化。

### KV-cache 和 Backend 缓存

Python backend 缓存以下 per-layer 稳定对象：

- KV pool tensor。
- Block table tensor。
- Pool index。
- 当前 KV-cache manager 和 quantization config 对应的 backend 对象。

backend cache 不只保存 `id()`，还保存并比较实际对象引用，避免 Python 对象释放后
`id()` 被复用导致拿到错误 backend。

### MLA Decode

删除临时 MLA decode workaround。最终实现直接调用 FlashInfer public API，不再传
手动 view 出来的 `out_buf`，也不再做 copy-back。

### NVTX 标记

为了定位剩余差距，在 trtllm-gen 和 `thop.attention` 两条路径都添加了 NVTX：

- trtllm-gen Python：context/generation preprocess、FlashInfer 调用、
  KV metadata、context postprocess。
- trtllm-gen C++：workspace materialization、decoder-info build、
  QKV preprocess、KV-cache postprocess、workspace zeroing。
- `thop.attention` C++：run/enqueue 阶段、metadata 构建、参数构造。

## 验证

新增 unit test：`cpp/tests/unit_tests/common/attentionWorkspaceTest.cpp`，覆盖：

- Context layout 顺序匹配原始 `AttentionOp` 顺序。
- Context materialization 返回正确 typed pointer，zero-size slice 返回 `nullptr`。
- Generation layout 中 CP workspace 位于 partial buffers 之前。
- XQA layout 使用单个 sparse-cache slice，匹配原始 workspace upper-bound 模型。
- FlashMLA layout 保持 accumulator 顺序。

rebase 到 #12525 后做过以下检查：

```bash
git diff --check fork/yihwang/shared_paged_index...HEAD
```

并对相关源码目录做了严格 conflict-marker 扫描。

调优过程中主要使用的 functional integration test：

```bash
TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 \
/usr/bin/python -m pytest \
  ./tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestGPTOSS::test_w4_1gpu[v1_kv_cache-True-True-trtllm-auto] \
  -s -v
```

fused preprocess 优化后的观察结果：

- Accuracy: 90.523。
- Test passed。
- trtllm-gen 优化后 TRTLLM execution time 大约为 53 秒。

本 session 早期同一 GPT-OSS workload 的结果：

- `thop.attention`: 45.678 秒 TRTLLM execution time。
- C++ hot-path cleanup 前的 trtllm-gen: 57.183 秒 TRTLLM execution time。

这些整测时间只适合做方向性判断，因为不同 capture 中 request mix 和 instance
count 会有差异。下面的 NVTX per-range 数据更适合分析具体瓶颈。

## Profiling 设置

nsys 数据在 B300 TensorRT-LLM docker 环境中收集。为了覆盖 steady-state 阶段，
使用 delayed capture：

- Delay: 85 秒。
- Duration: 45 秒。
- trtllm-gen 路径：`TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1`。
- 对比路径：默认 `thop.attention`。

`.sqlite` 和 `.stats.txt` 是本地 profiling artifact，不应提交。下面表格只摘录
关键 NVTX 和 CUDA API rows。

## 最终 trtllm-gen Profile

来源：`nsys_trtllm_gen_gptoss_cppnvtx_workspace_accessor_delay85_dur45.stats.txt`

| Range | Instances | Avg us | Median us | Total ms | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `TensorRT-LLM:trtllm_gen.generation.preprocess` | 18,074 | 31.553 | 33.902 | 570.289 | Python generation preprocess 外层 |
| `:trtllm_gen.cpp.generation_preprocess` | 18,074 | 22.132 | 24.796 | 400.006 | fused C++ generation preprocess |
| `TensorRT-LLM:trtllm_gen.generation.flashinfer` | 18,074 | 31.077 | 30.728 | 561.693 | FlashInfer public API 调用范围 |
| `TensorRT-LLM:trtllm_gen.generation.kv_metadata` | 18,074 | 8.459 | 8.063 | 152.892 | Python cached KV metadata lookup/slice |
| `:trtllm_gen.cpp.generation.zero_workspace` | 18,074 | 7.312 | 8.856 | 132.151 | FMHA 前 workspace zeroing |
| `:trtllm_gen.cpp.generation.materialize_workspace` | 18,074 | 5.340 | 5.905 | 96.509 | tensor view/raw pointer materialization |
| `TensorRT-LLM:trtllm_gen.context.preprocess` | 4,967 | 49.173 | 44.239 | 244.242 | Python context preprocess 外层 |
| `:trtllm_gen.cpp.context_preprocess` | 4,967 | 34.881 | 33.797 | 173.254 | fused C++ context preprocess |
| `:trtllm_gen.cpp.qkv_preprocess` | 23,041 | 7.462 | 7.695 | 171.937 | shared QKV preprocess 内层 |
| `TensorRT-LLM:trtllm_gen.context.flashinfer` | 4,967 | 33.343 | 30.005 | 165.613 | FlashInfer context 调用范围 |
| `TensorRT-LLM:trtllm_gen.context.kv_metadata` | 4,967 | 8.482 | 7.852 | 42.129 | Python cached KV metadata lookup/slice |
| `:trtllm_gen.cpp.context.zero_workspace` | 4,967 | 8.773 | 8.469 | 43.574 | FMHA 前 workspace zeroing |
| `:trtllm_gen.cpp.context.materialize_workspace` | 4,967 | 8.172 | 7.905 | 40.590 | tensor view/raw pointer materialization |
| `:trtllm_gen.cpp.context_postprocess` | 4,967 | 2.581 | 2.235 | 12.818 | fused C++ context postprocess 外层 |
| `:trtllm_gen.cpp.kv_cache_postprocess` | 4,967 | 1.626 | 1.405 | 8.076 | KV-cache postprocess 内层 |
| `:trtllm_gen.cpp.context_post.materialize_workspace` | 4,967 | 0.555 | 0.448 | 2.756 | postprocess view materialization |
| `cudaMemsetAsync` | 42,489 | 5.068 | 4.695 | 215.334 | CUDA API summary |

## thop.attention 对比 Profile

来源：`nsys_thop_gptoss_cppnvtx_run2_delay85_dur45.stats.txt`

| Range | Instances | Avg us | Median us | Total ms | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `:thop.attention` | 20,818 | 71.256 | 61.915 | 1,483.415 | 整个 thop attention op |
| `:thop.attention.run.generation` | 20,495 | 43.013 | 46.310 | 881.558 | generation run wrapper |
| `:thop.attention.enqueue_generation` | 20,495 | 26.817 | 28.423 | 549.605 | C++ enqueue generation |
| `:thop.attention.build_generation_params` | 20,495 | 0.367 | 0.288 | 7.520 | 参数构造 |
| `:thop.attention.run.context` | 6,011 | 54.523 | 53.217 | 327.735 | context run wrapper |
| `:thop.attention.enqueue_context` | 6,011 | 36.489 | 35.523 | 219.332 | C++ enqueue context |
| `:thop.attention.build_context_params` | 6,011 | 0.307 | 0.302 | 1.847 | 参数构造 |
| `:thop.attention.kv_metadata` | 26,506 | 6.646 | 6.907 | 176.160 | KV metadata |
| `:thop.attention.length_metadata` | 26,506 | 5.731 | 6.179 | 151.910 | length metadata |
| `cudaMemsetAsync` | 19,515 | 3.370 | 1.857 | 65.771 | CUDA API summary |

数据说明：FMHA kernel 本身并不能解释主要差距。更明显的差异在 host-side setup 和
workspace 管理：

- trtllm-gen 仍然有每次调用的 tensor view/raw pointer materialization 开销；
  `thop.attention` 则在 C++ 中直接构造 params。
- 最终 profile 中 trtllm-gen 的 `cudaMemsetAsync` 更多、总耗时更高：
  42,489 calls / 215.334 ms；`thop.attention` 是 19,515 calls / 65.771 ms。
- trtllm-gen workspace zeroing 仍然可见：generation 约 7.3 us/call，
  context 约 8.8 us/call。

## 优化迭代数据

下表跟踪各轮 trtllm-gen profile 中的关键 NVTX rows。所有时间都是 per-instance
average，单位为微秒。

| Profile | Python generation preprocess | C++ generation preprocess | Generation materialize | Generation zero workspace | Python context preprocess | C++ context preprocess | Context materialize |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nsys_trtllm_gen_gptoss_nvtx_20260501_delay85_dur45` | 41.861 | N/A | N/A | N/A | 58.570 | N/A | N/A |
| `nsys_trtllm_gen_gptoss_cppnvtx_run2_delay85_dur45` | 45.433 | 35.443 | 10.409 | 12.623 | 62.083 | 46.687 | 11.746 |
| `nsys_trtllm_gen_gptoss_cppnvtx_cuda_memset_delay85_dur45` | 39.923 | 30.205 | 10.195 | 7.608 | 58.029 | 41.611 | 11.790 |
| `nsys_trtllm_gen_gptoss_cppnvtx_rawptr_delay85_dur45` | 35.348 | 25.598 | 5.876 | 7.717 | 55.971 | 38.964 | 8.945 |
| `nsys_trtllm_gen_gptoss_cppnvtx_rawptr2_delay85_dur45` | 34.494 | 24.663 | 6.054 | 7.351 | 52.637 | 37.454 | 9.067 |
| `nsys_trtllm_gen_gptoss_cppnvtx_flatten_delay85_dur45` | 32.691 | 22.943 | 5.801 | 7.324 | 51.103 | 35.420 | 8.269 |
| `nsys_trtllm_gen_gptoss_cppnvtx_layoutcache_delay85_dur45` | 32.796 | 22.589 | 5.752 | 7.318 | 53.375 | 35.506 | 8.319 |
| `nsys_trtllm_gen_gptoss_cppnvtx_workspace_accessor_delay85_dur45` | 31.553 | 22.132 | 5.340 | 7.312 | 49.173 | 34.881 | 8.172 |

主要收益来自 raw-pointer workspace access 和 flatten C++ 内部调用链。layout cache
没有稳定收益，因此最终没有把它作为独立优化保留。

## 尝试过但没有保留的方案

- TVM FFI：context/generation dispatch 改成 TVM FFI 后没有看到有效收益，但代码
  复杂度明显增加，因此放弃。
- Workspace layout cache：收益不稳定，且相比周围 Python/C++ 和 workspace
  materialization 开销不明显，因此最终保留更简单的直接 C++ layout call。
- CUDA stream cache：没有采用，因为 current stream 是 PyTorch 状态，可能跨调用变化。
- 临时 FlashInfer MLA `out_buf` copy workaround：已删除，改回 FlashInfer public API。

## 希望 Maintainer 支持的理由

当前 PR 即使还没有完全追平 `thop.attention`，也值得支持：

- 消除了重复 workspace layout 逻辑，让 `AttentionOp` layout 可以被 trtllm-gen 复用。
- 删除 trtllm-gen hot path 中的 unfused Python/torch-op chain。
- 缓存稳定的 per-layer KV metadata，避免重复 metadata build。
- 在当前 workload 中，C++ generation preprocess 从约 35.4 us 降到 22.1 us；
  C++ context preprocess 从约 46.7 us 降到 34.9 us。
- 新增 NVTX 标记，让后续性能回归和剩余瓶颈更容易定位。

后续性能优化应重点关注：

- 研究 trtllm-gen workspace zeroing 是否可以安全减少或去掉。`thop.attention`
  没有显示同等规模的开销。
- 减少或绕过调用 FlashInfer 前的 per-call Tensor view materialization。
- 继续对比 FlashInfer public API 路径和直接 `AttentionOp` enqueue 路径。FMHA
  kernel generator 是共享的，剩余差距更可能来自 host-side setup，而不是 kernel math。
