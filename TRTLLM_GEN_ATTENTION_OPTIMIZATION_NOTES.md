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

# trtllm-gen Attention Optimization Notes

This note summarizes the trtllm-gen attention refactor and profiling work from
the optimization session. The goal was to make the trtllm-gen path easier to
maintain, reduce per-layer Python/C++ overhead, and explain the remaining gap
against the existing `thop.attention` path.

The current branch is stacked on #12525. That dependency is intentional because
#12525 changes the shared-paged-index behavior and KV-cache buffer calculation
used by the trtllm-gen FlashInfer path.

## Motivation

The original trtllm-gen attention integration had three problems:

1. Workspace layout logic was duplicated between `AttentionOp` and
   `trtllm_gen.py`, and `enqueueContext` / `enqueueGeneration` still manually
   split raw workspace pointers with repeated `nextWorkspacePtr` calls.
2. The trtllm-gen Python path rebuilt or re-materialized metadata every layer
   and every step, including KV-cache metadata that is stable for a layer once
   the KV pool and block table are known.
3. Profiling showed trtllm-gen was slower than `thop.attention` on the GPT-OSS
   W4 1-GPU test, even though both paths use trtllm-gen FMHA kernels generated
   from the same kernel generator. That made host-side overhead and workspace
   handling the main suspects.

## Implemented Changes

### Shared Attention Workspace Manager

Added `tensorrt_llm::common::op::AttentionWorkspaceManager` in
`cpp/tensorrt_llm/common/attentionWorkspace.h`.

It is now the single source of truth for these layouts:

- `AttentionOp` context workspace.
- `AttentionOp` generation workspace.
- FlashMLA workspace.
- XQA/trtllm-gen generation workspace.

The manager returns `WorkspaceSlice {offset, size}` layouts and can materialize
typed view structs from a raw workspace pointer. `AttentionOp` now consumes the
manager instead of open-coded pointer splitting. The trtllm-gen torch extension
uses the same layout ordering and exports the trtllm-gen subset to Python.

This directly removes the duplicated split logic that previously lived in
`enqueueContext` / `enqueueGeneration` and lets `thop.attention` and trtllm-gen
share the same workspace contract.

### trtllm_gen.py Refactor

`trtllm_gen.py` was reorganized around an object-oriented backend:

- `FlashInferTrtllmGenAttention` owns support checking, workspace layout,
  KV-cache metadata caching, and context/generation dispatch.
- `TrtllmAttentionWrapper` creates this backend only when
  `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1`.
- The backend is cached per `(kv_cache_manager, quant_config)` object identity.
- The old unfused trtllm-gen fallback path and unused torch op fakes were
  removed.
- The Python fallback workspace layout implementation was removed so the
  nanobind/C++ workspace manager is the only layout source.

This makes the Python side easier to extend: fused trtllm-gen support is now a
backend object instead of a collection of free functions spread across the file.

### Fused Preprocess/Postprocess Ops

Added internal fused torch-extension entry points:

- `trtllm_gen_context_preprocess`
- `trtllm_gen_context_postprocess`
- `trtllm_gen_generation_preprocess`

These replace the older per-step chain:

- `torch.ops.trtllm.build_decoder_info`
- `qkv_preprocessing`
- `kv_cache_postprocessing`

The fused ops take a `need_build_kv_cache_metadata` flag. Existing binding
defaults remain backward-compatible, but the Python backend passes
`need_build_kv_cache_metadata=False` after it has cached per-layer
`kv_cache`, `block_tables`, and pool index metadata. This avoids rebuilding
FlashInfer paged-KV metadata and avoids the previous host synchronization point
from reading pool mapping inside C++.

### C++ Hot-Path Cleanup

The fused preprocessing code is now treated as an internal C++ path instead of
a public torch op wrapper. The following overhead was removed or reduced:

- `BuildDecoderInfoParams` construction was inlined into the fused functions.
- Wrapper layers such as `build_decoder_info`, `invokeBuildDecoderInfoTyped`,
  and `qkv_processing<true/false>` were removed where they only forwarded to
  one internal call site.
- Workspace materialization now uses a `WorkspaceAccessor` that caches the base
  workspace pointer, tensor options, and element-size calculations.
- `makeOptionalWorkspaceTensorView` was removed. The code now creates only the
  tensor views that are actually required by FlashInfer or by the fused
  kernels.
- CUDA stream lookup remains local per fused op call. Cross-call stream caching
  was not used because the current stream can change under PyTorch.

### KV-Cache and Backend Caches

The Python backend caches these stable per-layer objects:

- KV pool tensor.
- Block table tensor.
- Pool index.
- Backend object for the active KV-cache manager and quantization config.

The backend cache stores and compares the actual object references, not only
`id()` values, to avoid Python `id()` reuse after garbage collection.

### MLA Decode

The temporary MLA decode workaround that passed a manually viewed `out_buf` and
copied the returned tensor back was removed. The final version calls the
FlashInfer public API directly and does not use `out_buf`.

### NVTX Instrumentation

NVTX ranges were added to both paths so the comparison is visible in Nsight
Systems:

- trtllm-gen Python ranges around context/generation preprocess, FlashInfer
  calls, KV metadata, and context postprocess.
- trtllm-gen C++ ranges around workspace materialization, decoder-info build,
  QKV preprocessing, KV-cache postprocess, and workspace zeroing.
- `thop.attention` C++ ranges around run/enqueue phases, metadata building, and
  parameter construction.

## Validation

The new workspace manager has unit coverage in
`cpp/tests/unit_tests/common/attentionWorkspaceTest.cpp`:

- Context layout ordering matches the original `AttentionOp` ordering.
- Context materialization returns typed pointers and `nullptr` for zero-size
  slices.
- Generation layout places CP workspace before partial buffers.
- XQA layout uses one sparse-cache slice, matching the original upper-bound
  workspace model.
- FlashMLA layout preserves accumulator order.

The branch was also validated during the session with:

```bash
git diff --check fork/yihwang/shared_paged_index...HEAD
```

and a strict conflict-marker scan after rebasing on #12525.

The main functional integration test used during tuning was:

```bash
TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 \
/usr/bin/python -m pytest \
  ./tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestGPTOSS::test_w4_1gpu[v1_kv_cache-True-True-trtllm-auto] \
  -s -v
```

Observed test result after the fused preprocessing optimizations:

- Accuracy: 90.523.
- Test passed.
- TRTLLM execution time was approximately 53 seconds in the optimized
  trtllm-gen run from this session.

Earlier in the session, the same GPT-OSS workload showed:

- `thop.attention`: 45.678 seconds TRTLLM execution time.
- trtllm-gen before the C++ hot-path cleanup: 57.183 seconds TRTLLM execution
  time.

These whole-test numbers are useful directionally, but the per-range NVTX data
below is more actionable because the request mix and instance counts can vary
between captures.

## Profiling Setup

Profiles were collected on B300 in the TensorRT-LLM docker environment with
Nsight Systems. The captures used delayed collection around the steady-state
part of the GPT-OSS W4 1-GPU test:

- Delay: 85 seconds.
- Duration: 45 seconds.
- trtllm-gen path: `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1`.
- Comparison path: default `thop.attention`.

The raw `.sqlite` and `.stats.txt` files are local profiling artifacts and are
not intended to be committed. The tables below summarize the relevant NVTX rows.

## Final trtllm-gen Profile

Source: `nsys_trtllm_gen_gptoss_cppnvtx_workspace_accessor_delay85_dur45.stats.txt`

| Range | Instances | Avg us | Median us | Total ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `TensorRT-LLM:trtllm_gen.generation.preprocess` | 18,074 | 31.553 | 33.902 | 570.289 | Python-level generation preprocess wrapper |
| `:trtllm_gen.cpp.generation_preprocess` | 18,074 | 22.132 | 24.796 | 400.006 | Fused C++ generation preprocess |
| `TensorRT-LLM:trtllm_gen.generation.flashinfer` | 18,074 | 31.077 | 30.728 | 561.693 | FlashInfer public API call range |
| `TensorRT-LLM:trtllm_gen.generation.kv_metadata` | 18,074 | 8.459 | 8.063 | 152.892 | Python cached KV metadata lookup/slice |
| `:trtllm_gen.cpp.generation.zero_workspace` | 18,074 | 7.312 | 8.856 | 132.151 | Workspace zeroing before FMHA |
| `:trtllm_gen.cpp.generation.materialize_workspace` | 18,074 | 5.340 | 5.905 | 96.509 | Tensor view/raw pointer materialization |
| `TensorRT-LLM:trtllm_gen.context.preprocess` | 4,967 | 49.173 | 44.239 | 244.242 | Python-level context preprocess wrapper |
| `:trtllm_gen.cpp.context_preprocess` | 4,967 | 34.881 | 33.797 | 173.254 | Fused C++ context preprocess |
| `:trtllm_gen.cpp.qkv_preprocess` | 23,041 | 7.462 | 7.695 | 171.937 | Shared QKV preprocessing inner range |
| `TensorRT-LLM:trtllm_gen.context.flashinfer` | 4,967 | 33.343 | 30.005 | 165.613 | FlashInfer context call range |
| `TensorRT-LLM:trtllm_gen.context.kv_metadata` | 4,967 | 8.482 | 7.852 | 42.129 | Python cached KV metadata lookup/slice |
| `:trtllm_gen.cpp.context.zero_workspace` | 4,967 | 8.773 | 8.469 | 43.574 | Workspace zeroing before FMHA |
| `:trtllm_gen.cpp.context.materialize_workspace` | 4,967 | 8.172 | 7.905 | 40.590 | Tensor view/raw pointer materialization |
| `:trtllm_gen.cpp.context_postprocess` | 4,967 | 2.581 | 2.235 | 12.818 | Fused C++ context postprocess wrapper |
| `:trtllm_gen.cpp.kv_cache_postprocess` | 4,967 | 1.626 | 1.405 | 8.076 | KV-cache postprocess inner range |
| `:trtllm_gen.cpp.context_post.materialize_workspace` | 4,967 | 0.555 | 0.448 | 2.756 | Postprocess view materialization |
| `cudaMemsetAsync` | 42,489 | 5.068 | 4.695 | 215.334 | CUDA API summary |

## thop.attention Comparison Profile

Source: `nsys_thop_gptoss_cppnvtx_run2_delay85_dur45.stats.txt`

| Range | Instances | Avg us | Median us | Total ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `:thop.attention` | 20,818 | 71.256 | 61.915 | 1,483.415 | Whole thop attention op |
| `:thop.attention.run.generation` | 20,495 | 43.013 | 46.310 | 881.558 | Generation run wrapper |
| `:thop.attention.enqueue_generation` | 20,495 | 26.817 | 28.423 | 549.605 | C++ enqueue generation |
| `:thop.attention.build_generation_params` | 20,495 | 0.367 | 0.288 | 7.520 | Param construction |
| `:thop.attention.run.context` | 6,011 | 54.523 | 53.217 | 327.735 | Context run wrapper |
| `:thop.attention.enqueue_context` | 6,011 | 36.489 | 35.523 | 219.332 | C++ enqueue context |
| `:thop.attention.build_context_params` | 6,011 | 0.307 | 0.302 | 1.847 | Param construction |
| `:thop.attention.kv_metadata` | 26,506 | 6.646 | 6.907 | 176.160 | KV metadata range |
| `:thop.attention.length_metadata` | 26,506 | 5.731 | 6.179 | 151.910 | Length metadata range |
| `cudaMemsetAsync` | 19,515 | 3.370 | 1.857 | 65.771 | CUDA API summary |

The FMHA kernel itself does not explain the large gap. The more visible
differences are host-side setup and workspace handling:

- trtllm-gen still has per-call materialization overhead that does not exist in
  the same form in `thop.attention` because thop builds params directly in C++.
- trtllm-gen shows more `cudaMemsetAsync` calls and higher total memset time in
  the final profile: 42,489 calls / 215.334 ms versus 19,515 calls / 65.771 ms.
- Workspace zeroing remains visible in trtllm-gen: about 7.3 us per generation
  call and 8.8 us per context call.

## Optimization Iterations

The table tracks the main trtllm-gen NVTX rows across profiling iterations. All
times are per-instance averages in microseconds.

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

The useful wins came from raw-pointer workspace access and flattening the
internal C++ call chain. Layout caching did not provide a meaningful gain and
was not kept as a standalone optimization.

## Things Tried and Not Kept

- TVM FFI was considered for context/generation dispatch. It did not show a
  useful benefit for this workload and would make the code significantly more
  complex, so it was dropped.
- Workspace-layout caching was tested. It produced no stable gain compared with
  the cost of the surrounding Python/C++ and workspace materialization path, so
  the final code keeps the simpler direct C++ layout call.
- CUDA stream caching was considered. It was not used because the current stream
  is PyTorch state and can change across calls.
- The temporary FlashInfer MLA output-buffer copy workaround was removed in
  favor of the public FlashInfer API.

## Maintainer Ask

The current PR is worth supporting even though it does not fully close the
thop.attention gap:

- It removes duplicated workspace layout logic and makes the `AttentionOp`
  layout reusable by trtllm-gen.
- It removes the unfused Python/torch-op chain from the trtllm-gen hot path.
- It caches stable per-layer KV metadata and avoids repeated metadata rebuilds.
- It reduces C++ generation preprocess from roughly 35.4 us to 22.1 us in the
  profiled workload, and C++ context preprocess from roughly 46.7 us to 34.9 us.
- It adds NVTX ranges that make future regressions and remaining overhead much
  easier to attribute.

The remaining performance work should focus on the parts still visible in NVTX:

- Understand whether trtllm-gen workspace zeroing can be reduced or eliminated
  safely. `thop.attention` does not show an equivalent cost at the same scale.
- Reduce or bypass per-call Tensor view materialization when calling FlashInfer.
- Continue comparing the public FlashInfer path against the direct
  `AttentionOp` enqueue path, because the FMHA kernel generator is shared and
  the remaining gap is likely in host-side setup rather than kernel math.
