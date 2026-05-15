/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bindings.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <tensorrt_llm/kernels/helixAllToAll.h>
#include <tensorrt_llm/thop/attentionOp.h>
#include <tensorrt_llm/thop/moeAlltoAllMeta.h>
#include <tensorrt_llm/thop/outputTensor.h>
#include <tensorrt_llm/thop/trtllmGenFusedOps.h>
#include <torch/extension.h>

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::thop
{

namespace
{

template <typename T>
nb::object optionalToObject(std::optional<T> const& value)
{
    if (value.has_value())
    {
        return nb::cast(*value);
    }
    return nb::none();
}

nb::tuple trtllmGenContextPreprocessBinding(torch::Tensor qkv_input, torch::Tensor workspace,
    torch::Tensor sequence_lengths, torch::Tensor context_lengths, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> attention_output_orig_quant, std::optional<torch::Tensor> rotary_inv_freq,
    std::optional<torch::Tensor> rotary_cos_sin, std::optional<torch::Tensor> mrope_rotary_cos_sin, int64_t layer_idx,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_size, int64_t tokens_per_block, int64_t mask_type,
    int64_t kv_cache_quant_mode, int64_t max_attention_window_size, int64_t cyclic_attention_window_size,
    int64_t sink_token_length, int64_t num_tokens, int64_t batch_size, int64_t input_seq_length,
    int64_t max_past_kv_length, int64_t rotary_embedding_dim, double rotary_embedding_base,
    int64_t rotary_embedding_scale_type, double rotary_embedding_scale, int64_t rotary_embedding_max_positions,
    int64_t position_embedding_type, double bmm1_scale, double bmm2_scale, int64_t attention_chunk_size,
    bool fp8_context_fmha, bool paged_context_fmha, bool is_mla_enable, int64_t multi_processor_count,
    int64_t total_num_blocks, int64_t kv_factor, bool need_build_kv_cache_metadata)
{
    auto result = [&]()
    {
        nb::gil_scoped_release release;
        return torch_ext::trtllmGenContextPreprocess(qkv_input, workspace, sequence_lengths, context_lengths,
            kv_cache_block_offsets, host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, kv_scale_orig_quant,
            kv_scale_quant_orig, attention_output_orig_quant, rotary_inv_freq, rotary_cos_sin, mrope_rotary_cos_sin,
            layer_idx, num_heads, num_kv_heads, head_size, tokens_per_block, mask_type, kv_cache_quant_mode,
            max_attention_window_size, cyclic_attention_window_size, sink_token_length, num_tokens, batch_size,
            input_seq_length, max_past_kv_length, rotary_embedding_dim, rotary_embedding_base,
            rotary_embedding_scale_type, rotary_embedding_scale, rotary_embedding_max_positions,
            position_embedding_type, bmm1_scale, bmm2_scale, attention_chunk_size, fp8_context_fmha, paged_context_fmha,
            is_mla_enable, multi_processor_count, total_num_blocks, kv_factor, need_build_kv_cache_metadata);
    }();

    return nb::make_tuple(std::get<0>(result), optionalToObject(std::get<1>(result)),
        optionalToObject(std::get<2>(result)), std::get<3>(result), std::get<4>(result), std::get<5>(result),
        std::get<6>(result), std::get<7>(result), std::get<8>(result));
}

nb::tuple trtllmGenGenerationPreprocessBinding(torch::Tensor qkv_input, torch::Tensor workspace,
    torch::Tensor sequence_lengths, std::optional<torch::Tensor> spec_decoding_generation_lengths,
    std::optional<torch::Tensor> spec_decoding_position_offsets, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> attention_output_orig_quant, std::optional<torch::Tensor> rotary_inv_freq,
    std::optional<torch::Tensor> rotary_cos_sin, int64_t layer_idx, int64_t seq_offset, int64_t num_heads,
    int64_t num_kv_heads, int64_t head_size, int64_t tokens_per_block, int64_t kv_cache_quant_mode,
    int64_t max_attention_window_size, int64_t cyclic_attention_window_size, int64_t sink_token_length,
    int64_t num_tokens, int64_t batch_beam, int64_t input_seq_length, int64_t max_past_kv_length,
    int64_t rotary_embedding_dim, double rotary_embedding_base, int64_t rotary_embedding_scale_type,
    double rotary_embedding_scale, int64_t rotary_embedding_max_positions, int64_t position_embedding_type,
    double bmm1_scale, double bmm2_scale, bool fp8_context_fmha, int64_t predicted_tokens_per_seq,
    int64_t attention_chunk_size, int64_t multi_processor_count, int64_t total_num_blocks, int64_t kv_factor,
    bool need_build_kv_cache_metadata)
{
    auto result = [&]()
    {
        nb::gil_scoped_release release;
        return torch_ext::trtllmGenGenerationPreprocess(qkv_input, workspace, sequence_lengths,
            spec_decoding_generation_lengths, spec_decoding_position_offsets, kv_cache_block_offsets,
            host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, kv_scale_orig_quant, kv_scale_quant_orig,
            attention_output_orig_quant, rotary_inv_freq, rotary_cos_sin, layer_idx, seq_offset, num_heads,
            num_kv_heads, head_size, tokens_per_block, kv_cache_quant_mode, max_attention_window_size,
            cyclic_attention_window_size, sink_token_length, num_tokens, batch_beam, input_seq_length,
            max_past_kv_length, rotary_embedding_dim, rotary_embedding_base, rotary_embedding_scale_type,
            rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, bmm1_scale, bmm2_scale,
            fp8_context_fmha, predicted_tokens_per_seq, attention_chunk_size, multi_processor_count, total_num_blocks,
            kv_factor, need_build_kv_cache_metadata);
    }();

    return nb::make_tuple(std::get<0>(result), optionalToObject(std::get<1>(result)),
        optionalToObject(std::get<2>(result)), std::get<3>(result), optionalToObject(std::get<4>(result)),
        std::get<5>(result), std::get<6>(result), std::get<7>(result), std::get<8>(result));
}

} // namespace

void initBindings(nb::module_& m)
{
    // Sync with torch_ext::BufferKind in tensorrt_llm/thop/outputTensor.h
    nb::enum_<torch_ext::BufferKind>(m, "BufferKind", nb::is_arithmetic())
        .value("DEFAULT", torch_ext::BufferKind::Default)
        .value("USERBUFFERS", torch_ext::BufferKind::Userbuffers)
        .value("NCCL_WINDOW", torch_ext::BufferKind::NcclWindow);

    // Export MoE A2A constants
    for (auto const& kv : torch_ext::moe_comm::getMoeA2AMetaInfoIndexPairs())
    {
        m.attr(kv.first) = kv.second;
    }

    nb::enum_<torch_ext::AttentionInputType>(m, "AttentionInputType", nb::is_arithmetic())
        .value("Mixed", torch_ext::AttentionInputType::Mixed)
        .value("ContextOnly", torch_ext::AttentionInputType::ContextOnly)
        .value("GenerationOnly", torch_ext::AttentionInputType::GenerationOnly);

    nb::class_<torch_ext::TrtllmRopeArgs>(m, "TrtllmRopeArgs")
        .def(nb::init<int64_t, int64_t, double, int64_t, std::vector<double>, std::vector<int64_t>,
                 std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
                 std::optional<at::Tensor>>(),
            nb::arg("position_embedding_type"), nb::arg("rotary_embedding_dim"), nb::arg("rotary_embedding_base"),
            nb::arg("rotary_embedding_scale_type"), nb::arg("rotary_embedding_scales"),
            nb::arg("rotary_embedding_max_position_info"), nb::arg("rotary_inv_freq").none(),
            nb::arg("rotary_cos_sin").none(), nb::arg("mrope_rotary_cos_sin") = std::nullopt,
            nb::arg("mrope_position_deltas") = std::nullopt)
        .def_ro("position_embedding_type", &torch_ext::TrtllmRopeArgs::positionEmbeddingType)
        .def_ro("rotary_embedding_dim", &torch_ext::TrtllmRopeArgs::rotaryEmbeddingDim)
        .def_ro("rotary_embedding_base", &torch_ext::TrtllmRopeArgs::rotaryEmbeddingBase)
        .def_ro("rotary_embedding_scale_type", &torch_ext::TrtllmRopeArgs::rotaryEmbeddingScaleType)
        .def_ro("rotary_embedding_scales", &torch_ext::TrtllmRopeArgs::rotaryEmbeddingScales)
        .def_ro("rotary_embedding_max_position_info", &torch_ext::TrtllmRopeArgs::rotaryEmbeddingMaxPositionInfo)
        .def_ro("rotary_inv_freq", &torch_ext::TrtllmRopeArgs::rotaryInvFreq)
        .def_ro("rotary_cos_sin", &torch_ext::TrtllmRopeArgs::rotaryCosSin)
        .def_rw("mrope_rotary_cos_sin", &torch_ext::TrtllmRopeArgs::mropeRotaryCosSin)
        .def_rw("mrope_position_deltas", &torch_ext::TrtllmRopeArgs::mropePositionDeltas);

    nb::class_<torch_ext::TrtllmQuantArgs>(m, "TrtllmQuantArgs")
        .def(nb::init<std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>>(),
            nb::arg("kv_scale_orig_quant") = std::nullopt, nb::arg("kv_scale_quant_orig") = std::nullopt,
            nb::arg("out_scale") = std::nullopt)
        .def_rw("kv_scale_orig_quant", &torch_ext::TrtllmQuantArgs::kvScaleOrigQuant)
        .def_rw("kv_scale_quant_orig", &torch_ext::TrtllmQuantArgs::kvScaleQuantOrig)
        .def_rw("out_scale", &torch_ext::TrtllmQuantArgs::outScale);

    nb::class_<torch_ext::TrtllmFmhaArgs>(m, "TrtllmFmhaArgs")
        .def(nb::init<std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
                 std::optional<at::Tensor>, std::optional<at::Tensor>, int64_t>(),
            nb::arg("attention_sinks") = std::nullopt, nb::arg("softmax_stats_tensor") = std::nullopt,
            nb::arg("cu_q_seqlens") = std::nullopt, nb::arg("cu_kv_seqlens") = std::nullopt,
            nb::arg("fmha_scheduler_counter") = std::nullopt, nb::arg("chunked_prefill_buffer_batch_size") = 1)
        .def_rw("attention_sinks", &torch_ext::TrtllmFmhaArgs::attentionSinks)
        .def_rw("softmax_stats_tensor", &torch_ext::TrtllmFmhaArgs::softmaxStatsTensor)
        .def_rw("cu_q_seqlens", &torch_ext::TrtllmFmhaArgs::cuQSeqlens)
        .def_rw("cu_kv_seqlens", &torch_ext::TrtllmFmhaArgs::cuKvSeqlens)
        .def_rw("fmha_scheduler_counter", &torch_ext::TrtllmFmhaArgs::fmhaSchedulerCounter)
        .def_rw("chunked_prefill_buffer_batch_size", &torch_ext::TrtllmFmhaArgs::chunkedPrefillBufferBatchSize);

    nb::class_<torch_ext::TrtllmKvCacheArgs>(m, "TrtllmKvCacheArgs")
        .def(nb::init<std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
                 std::optional<at::Tensor>, int64_t, nvinfer1::DataType, int64_t, int64_t, std::optional<int64_t>>(),
            nb::arg("block_offsets").none(), nb::arg("host_pool_pointers").none(), nb::arg("host_pool_mapping").none(),
            nb::arg("cache_indirection").none(), nb::arg("tokens_per_block"), nb::arg("dtype"), nb::arg("kv_factor"),
            nb::arg("total_num_blocks"), nb::arg("compressed_kv_cache_pool_ptr") = std::nullopt)
        .def_rw("block_offsets", &torch_ext::TrtllmKvCacheArgs::blockOffsets)
        .def_rw("host_pool_pointers", &torch_ext::TrtllmKvCacheArgs::hostPoolPointers)
        .def_rw("host_pool_mapping", &torch_ext::TrtllmKvCacheArgs::hostPoolMapping)
        .def_rw("cache_indirection", &torch_ext::TrtllmKvCacheArgs::cacheIndirection)
        .def_ro("tokens_per_block", &torch_ext::TrtllmKvCacheArgs::tokensPerBlock)
        .def_ro("dtype", &torch_ext::TrtllmKvCacheArgs::dtype)
        .def_ro("kv_factor", &torch_ext::TrtllmKvCacheArgs::kvFactor)
        .def_ro("total_num_blocks", &torch_ext::TrtllmKvCacheArgs::totalNumBlocks)
        .def_rw("compressed_kv_cache_pool_ptr", &torch_ext::TrtllmKvCacheArgs::compressedKvCachePoolPtr);

    nb::class_<torch_ext::TrtllmMlaArgs>(m, "TrtllmMlaArgs")
        .def(nb::init<std::optional<int64_t>, int64_t, int64_t, int64_t, int64_t, bool, std::optional<at::Tensor>,
                 std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
                 std::optional<at::Tensor>>(),
            nb::arg("q_lora_rank").none(), nb::arg("kv_lora_rank"), nb::arg("qk_nope_head_dim"),
            nb::arg("qk_rope_head_dim"), nb::arg("v_head_dim"), nb::arg("rope_append"),
            nb::arg("latent_cache") = std::nullopt, nb::arg("q_pe") = std::nullopt,
            nb::arg("mla_bmm1_scale") = std::nullopt, nb::arg("mla_bmm2_scale") = std::nullopt,
            nb::arg("quant_q_buffer") = std::nullopt)
        .def_ro("q_lora_rank", &torch_ext::TrtllmMlaArgs::qLoraRank)
        .def_ro("kv_lora_rank", &torch_ext::TrtllmMlaArgs::kvLoraRank)
        .def_ro("qk_nope_head_dim", &torch_ext::TrtllmMlaArgs::qkNopeHeadDim)
        .def_ro("qk_rope_head_dim", &torch_ext::TrtllmMlaArgs::qkRopeHeadDim)
        .def_ro("v_head_dim", &torch_ext::TrtllmMlaArgs::vHeadDim)
        .def_ro("rope_append", &torch_ext::TrtllmMlaArgs::ropeAppend)
        .def_rw("latent_cache", &torch_ext::TrtllmMlaArgs::latentCache)
        .def_rw("q_pe", &torch_ext::TrtllmMlaArgs::qPe)
        .def_rw("mla_bmm1_scale", &torch_ext::TrtllmMlaArgs::mlaBmm1Scale)
        .def_rw("mla_bmm2_scale", &torch_ext::TrtllmMlaArgs::mlaBmm2Scale)
        .def_rw("quant_q_buffer", &torch_ext::TrtllmMlaArgs::quantQBuffer);

    nb::class_<torch_ext::TrtllmSageArgs>(m, "TrtllmSageArgs")
        .def(nb::init<int64_t, int64_t, int64_t, bool>(), nb::arg("num_elts_per_blk_q"), nb::arg("num_elts_per_blk_k"),
            nb::arg("num_elts_per_blk_v"), nb::arg("qk_int8"))
        .def_rw("num_elts_per_blk_q", &torch_ext::TrtllmSageArgs::numEltsPerBlkQ)
        .def_rw("num_elts_per_blk_k", &torch_ext::TrtllmSageArgs::numEltsPerBlkK)
        .def_rw("num_elts_per_blk_v", &torch_ext::TrtllmSageArgs::numEltsPerBlkV)
        .def_rw("qk_int8", &torch_ext::TrtllmSageArgs::qkInt8);

    nb::class_<torch_ext::TrtllmSparseArgs>(m, "TrtllmSparseArgs")
        .def(nb::init<std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
                 std::optional<at::Tensor>, int64_t, std::optional<int64_t>, std::optional<at::Tensor>>(),
            nb::arg("sparse_kv_indices") = std::nullopt, nb::arg("sparse_kv_offsets") = std::nullopt,
            nb::arg("sparse_attn_indices") = std::nullopt, nb::arg("sparse_attn_offsets") = std::nullopt,
            nb::arg("sparse_attn_indices_block_size") = 0, nb::arg("num_sparse_topk") = std::nullopt,
            nb::arg("sparse_mla_topk_lens") = std::nullopt)
        .def_rw("sparse_kv_indices", &torch_ext::TrtllmSparseArgs::sparseKvIndices)
        .def_rw("sparse_kv_offsets", &torch_ext::TrtllmSparseArgs::sparseKvOffsets)
        .def_rw("sparse_attn_indices", &torch_ext::TrtllmSparseArgs::sparseAttnIndices)
        .def_rw("sparse_attn_offsets", &torch_ext::TrtllmSparseArgs::sparseAttnOffsets)
        .def_rw("sparse_attn_indices_block_size", &torch_ext::TrtllmSparseArgs::sparseAttnIndicesBlockSize)
        .def_rw("num_sparse_topk", &torch_ext::TrtllmSparseArgs::numSparseTopk)
        .def_rw("sparse_mla_topk_lens", &torch_ext::TrtllmSparseArgs::sparseMlaTopkLens);

    nb::class_<torch_ext::TrtllmSkipSoftmaxArgs>(m, "TrtllmSkipSoftmaxArgs")
        .def(nb::init<std::optional<double>, std::optional<double>, std::optional<at::Tensor>>(),
            nb::arg("threshold_scale_factor_prefill") = std::nullopt,
            nb::arg("threshold_scale_factor_decode") = std::nullopt, nb::arg("block_skip_stat") = std::nullopt)
        .def_rw("threshold_scale_factor_prefill", &torch_ext::TrtllmSkipSoftmaxArgs::thresholdScaleFactorPrefill)
        .def_rw("threshold_scale_factor_decode", &torch_ext::TrtllmSkipSoftmaxArgs::thresholdScaleFactorDecode)
        .def_rw("block_skip_stat", &torch_ext::TrtllmSkipSoftmaxArgs::blockSkipStat);

    nb::class_<torch_ext::TrtllmSpecDecArgs>(m, "TrtllmSpecDecArgs")
        .def(nb::init<bool, bool, std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
                 std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>>(),
            nb::arg("use_spec_decoding"), nb::arg("is_spec_dec_tree"), nb::arg("generation_lengths") = std::nullopt,
            nb::arg("position_offsets") = std::nullopt, nb::arg("packed_mask") = std::nullopt,
            nb::arg("bl_tree_mask_offset") = std::nullopt, nb::arg("bl_tree_mask") = std::nullopt,
            nb::arg("first_sparse_mask_offset_kv") = std::nullopt)
        .def_rw("use_spec_decoding", &torch_ext::TrtllmSpecDecArgs::useSpecDecoding)
        .def_rw("is_spec_dec_tree", &torch_ext::TrtllmSpecDecArgs::isSpecDecTree)
        .def_rw("generation_lengths", &torch_ext::TrtllmSpecDecArgs::generationLengths)
        .def_rw("position_offsets", &torch_ext::TrtllmSpecDecArgs::positionOffsets)
        .def_rw("packed_mask", &torch_ext::TrtllmSpecDecArgs::packedMask)
        .def_rw("bl_tree_mask_offset", &torch_ext::TrtllmSpecDecArgs::blTreeMaskOffset)
        .def_rw("bl_tree_mask", &torch_ext::TrtllmSpecDecArgs::blTreeMask)
        .def_rw("first_sparse_mask_offset_kv", &torch_ext::TrtllmSpecDecArgs::firstSparseMaskOffsetKv);

    nb::class_<torch_ext::TrtllmHelixArgs>(m, "TrtllmHelixArgs")
        .def(nb::init<at::Tensor, at::Tensor>(), nb::arg("position_offsets"), nb::arg("is_inactive_rank"))
        .def_rw("position_offsets", &torch_ext::TrtllmHelixArgs::positionOffsets)
        .def_rw("is_inactive_rank", &torch_ext::TrtllmHelixArgs::isInactiveRank);

    nb::class_<torch_ext::TrtllmFlashMlaArgs>(m, "TrtllmFlashMlaArgs")
        .def(nb::init<at::Tensor, at::Tensor>(), nb::arg("tile_scheduler_metadata"), nb::arg("num_splits"))
        .def_rw("tile_scheduler_metadata", &torch_ext::TrtllmFlashMlaArgs::tileSchedulerMetadata)
        .def_rw("num_splits", &torch_ext::TrtllmFlashMlaArgs::numSplits);

    nb::class_<torch_ext::TrtllmAttentionArgs>(m, "TrtllmAttentionArgs")
        .def(nb::init<at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>, at::Tensor,
                 std::optional<at::Tensor>, at::Tensor, int64_t, int64_t, int64_t, int64_t, double, int64_t,
                 std::optional<int64_t>, int64_t, int64_t, int64_t, int64_t, int64_t, torch_ext::AttentionInputType,
                 bool, bool, bool, std::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                 at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, torch_ext::TrtllmRopeArgs,
                 torch_ext::TrtllmQuantArgs, torch_ext::TrtllmFmhaArgs, std::optional<torch_ext::TrtllmKvCacheArgs>,
                 std::optional<torch_ext::TrtllmMlaArgs>, std::optional<torch_ext::TrtllmSageArgs>,
                 std::optional<torch_ext::TrtllmSparseArgs>, std::optional<torch_ext::TrtllmSkipSoftmaxArgs>,
                 std::optional<torch_ext::TrtllmSpecDecArgs>, std::optional<torch_ext::TrtllmHelixArgs>,
                 std::optional<torch_ext::TrtllmFlashMlaArgs>>(),
            nb::arg("q"), nb::arg("k").none(), nb::arg("v").none(), nb::arg("output"), nb::arg("output_sf").none(),
            nb::arg("workspace"), nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"),
            nb::arg("predicted_tokens_per_seq"), nb::arg("q_scaling"), nb::arg("quant_mode"),
            nb::arg("attention_chunk_size").none(), nb::arg("sink_token_length"), nb::arg("layer_idx"),
            nb::arg("mask_type"), nb::arg("attention_window_size"), nb::arg("max_attention_window_size"),
            nb::arg("attention_input_type"), nb::arg("is_fused_qkv"), nb::arg("update_kv_cache"),
            nb::arg("use_paged_context_fmha"), nb::arg("block_ids_per_seq").none(), nb::arg("sequence_length"),
            nb::arg("host_past_key_value_lengths"), nb::arg("host_total_kv_lens"), nb::arg("context_lengths"),
            nb::arg("host_context_lengths"), nb::arg("host_request_types"), nb::arg("num_contexts"),
            nb::arg("num_ctx_tokens"), nb::arg("max_num_requests"), nb::arg("max_context_length"),
            nb::arg("beam_width"), nb::arg("rope"), nb::arg("quant"), nb::arg("fmha"),
            nb::arg("kv_cache") = std::nullopt, nb::arg("mla") = std::nullopt, nb::arg("sage") = std::nullopt,
            nb::arg("sparse") = std::nullopt, nb::arg("skip_softmax") = std::nullopt,
            nb::arg("spec_dec") = std::nullopt, nb::arg("helix") = std::nullopt, nb::arg("flash_mla") = std::nullopt)
        .def_rw("q", &torch_ext::TrtllmAttentionArgs::q)
        .def_rw("k", &torch_ext::TrtllmAttentionArgs::k)
        .def_rw("v", &torch_ext::TrtllmAttentionArgs::v)
        .def_rw("output", &torch_ext::TrtllmAttentionArgs::output)
        .def_rw("output_sf", &torch_ext::TrtllmAttentionArgs::outputSf)
        .def_rw("workspace", &torch_ext::TrtllmAttentionArgs::workspace)
        .def_ro("num_heads", &torch_ext::TrtllmAttentionArgs::numHeads)
        .def_ro("num_kv_heads", &torch_ext::TrtllmAttentionArgs::numKvHeads)
        .def_ro("head_size", &torch_ext::TrtllmAttentionArgs::headSize)
        .def_ro("predicted_tokens_per_seq", &torch_ext::TrtllmAttentionArgs::predictedTokensPerSeq)
        .def_ro("q_scaling", &torch_ext::TrtllmAttentionArgs::qScaling)
        .def_ro("quant_mode", &torch_ext::TrtllmAttentionArgs::quantMode)
        .def_ro("attention_chunk_size", &torch_ext::TrtllmAttentionArgs::attentionChunkSize)
        .def_ro("sink_token_length", &torch_ext::TrtllmAttentionArgs::sinkTokenLength)
        .def_rw("layer_idx", &torch_ext::TrtllmAttentionArgs::layerIdx)
        .def_rw("mask_type", &torch_ext::TrtllmAttentionArgs::maskType)
        .def_rw("attention_window_size", &torch_ext::TrtllmAttentionArgs::attentionWindowSize)
        .def_rw("max_attention_window_size", &torch_ext::TrtllmAttentionArgs::maxAttentionWindowSize)
        .def_rw("attention_input_type", &torch_ext::TrtllmAttentionArgs::attentionInputType)
        .def_rw("is_fused_qkv", &torch_ext::TrtllmAttentionArgs::isFusedQkv)
        .def_rw("update_kv_cache", &torch_ext::TrtllmAttentionArgs::updateKvCache)
        .def_rw("use_paged_context_fmha", &torch_ext::TrtllmAttentionArgs::usePagedContextFmha)
        .def_rw("block_ids_per_seq", &torch_ext::TrtllmAttentionArgs::blockIdsPerSeq)
        .def_rw("sequence_length", &torch_ext::TrtllmAttentionArgs::sequenceLength)
        .def_rw("host_past_key_value_lengths", &torch_ext::TrtllmAttentionArgs::hostPastKeyValueLengths)
        .def_rw("host_total_kv_lens", &torch_ext::TrtllmAttentionArgs::hostTotalKvLens)
        .def_rw("context_lengths", &torch_ext::TrtllmAttentionArgs::contextLengths)
        .def_rw("host_context_lengths", &torch_ext::TrtllmAttentionArgs::hostContextLengths)
        .def_rw("host_request_types", &torch_ext::TrtllmAttentionArgs::hostRequestTypes)
        .def_rw("num_contexts", &torch_ext::TrtllmAttentionArgs::numContexts)
        .def_rw("num_ctx_tokens", &torch_ext::TrtllmAttentionArgs::numCtxTokens)
        .def_rw("max_num_requests", &torch_ext::TrtllmAttentionArgs::maxNumRequests)
        .def_rw("max_context_length", &torch_ext::TrtllmAttentionArgs::maxContextLength)
        .def_rw("beam_width", &torch_ext::TrtllmAttentionArgs::beamWidth)
        .def_ro("rope", &torch_ext::TrtllmAttentionArgs::rope)
        .def_rw("quant", &torch_ext::TrtllmAttentionArgs::quant)
        .def_rw("fmha", &torch_ext::TrtllmAttentionArgs::fmha)
        .def_prop_ro("kv_cache",
            [](torch_ext::TrtllmAttentionArgs& args) -> std::optional<torch_ext::TrtllmKvCacheArgs>&
            { return args.kvCache; })
        .def_prop_ro("mla",
            [](torch_ext::TrtllmAttentionArgs& args) -> std::optional<torch_ext::TrtllmMlaArgs>& { return args.mla; })
        .def_prop_ro("sage",
            [](torch_ext::TrtllmAttentionArgs& args) -> std::optional<torch_ext::TrtllmSageArgs>& { return args.sage; })
        .def_prop_ro("sparse",
            [](torch_ext::TrtllmAttentionArgs& args) -> std::optional<torch_ext::TrtllmSparseArgs>&
            { return args.sparse; })
        .def_prop_ro("skip_softmax",
            [](torch_ext::TrtllmAttentionArgs& args) -> std::optional<torch_ext::TrtllmSkipSoftmaxArgs>&
            { return args.skipSoftmax; })
        .def_prop_ro("spec_dec",
            [](torch_ext::TrtllmAttentionArgs& args) -> std::optional<torch_ext::TrtllmSpecDecArgs>&
            { return args.specDec; })
        .def_prop_ro("helix",
            [](torch_ext::TrtllmAttentionArgs& args) -> std::optional<torch_ext::TrtllmHelixArgs>&
            { return args.helix; })
        .def_prop_ro("flash_mla",
            [](torch_ext::TrtllmAttentionArgs& args) -> std::optional<torch_ext::TrtllmFlashMlaArgs>&
            { return args.flashMla; });

    m.def("get_attention_workspace_size", &torch_ext::getAttentionWorkspaceSize, nb::arg("args"),
        "Return required thop attention workspace size in bytes.", nb::call_guard<nb::gil_scoped_release>());

    m.def("attention", &torch_ext::attention, nb::arg("args"), "Multi-head attention operation",
        nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "get_helix_workspace_size_per_rank",
        [](int cp_size) { return tensorrt_llm::kernels::computeHelixWorkspaceSizePerRank(cp_size); },
        nb::arg("cp_size"), "Get helix all-to-all workspace size per rank in bytes");

    m.def("compute_flash_mla_metadata", &tensorrt_llm::computeFlashMlaMetadata, nb::arg("seqlens_k"),
        nb::arg("tile_scheduler_metadata"), nb::arg("num_splits"), nb::arg("batch_size"), nb::arg("s_q"),
        nb::arg("num_q_heads"), nb::arg("num_kv_heads"), nb::arg("head_size_v"),
        "Compute FlashMLA tile-scheduler metadata in-place. Call once per forward pass before attention layers.",
        nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "get_trtllm_gen_context_workspace_layout",
        [](at::ScalarType dtype, int64_t batch_size, int64_t num_tokens, int64_t num_heads, int64_t head_size,
            int64_t rotary_embedding_dim, bool separate_q_kv_input, bool fp8_context_fmha)
        {
            auto const layout = torch_ext::TrtllmAttentionWorkspaceManager::buildContextLayout(dtype, batch_size,
                num_tokens, num_heads, head_size, rotary_embedding_dim, separate_q_kv_input, fp8_context_fmha);
            nb::dict result;
            result["trtllm_gen_workspace_offset"] = layout.trtllmGenWorkspaceOffset;
            result["cu_q_seqlens_offset"] = layout.cuQSeqlensOffset;
            result["cu_kv_seqlens_offset"] = layout.cuKvSeqlensOffset;
            result["cu_mask_rows_offset"] = layout.cuMaskRowsOffset;
            result["rotary_inv_freq_offset"] = layout.rotaryInvFreqOffset;
            result["q_buf_offset"] = layout.qBufOffset;
            result["tokens_info_offset"] = layout.tokensInfoOffset;
            result["fmha_tile_counter_offset"] = layout.fmhaTileCounterOffset;
            result["fmha_bmm1_scale_offset"] = layout.fmhaBmm1ScaleOffset;
            result["fmha_bmm2_scale_offset"] = layout.fmhaBmm2ScaleOffset;
            result["trtllm_gen_workspace_size"] = layout.trtllmGenWorkspaceSize;
            result["cu_seqlens_size"] = layout.cuSeqlensSize;
            result["rotary_inv_freq_size"] = layout.rotaryInvFreqSize;
            result["q_buf_size"] = layout.qBufSize;
            result["tokens_info_size"] = layout.tokensInfoSize;
            result["fmha_scheduler_counter_size"] = layout.fmhaTileCounterSize;
            result["fmha_bmm1_scale_size"] = layout.fmhaBmm1ScaleSize;
            result["fmha_bmm2_scale_size"] = layout.fmhaBmm2ScaleSize;
            result["total_size"] = layout.totalSize;
            return result;
        },
        nb::arg("dtype"), nb::arg("batch_size"), nb::arg("num_tokens"), nb::arg("num_heads"), nb::arg("head_size"),
        nb::arg("rotary_embedding_dim"), nb::arg("separate_q_kv_input"), nb::arg("fp8_context_fmha"),
        "Return the C++ trtllm-gen context workspace layout.");

    m.def(
        "get_trtllm_gen_generation_workspace_layout",
        [](at::ScalarType dtype, int64_t batch_beam, int64_t num_tokens, int64_t num_heads, int64_t head_size,
            int64_t rotary_embedding_dim, int64_t num_kv_heads, int64_t max_blocks_per_sequence,
            bool use_sparse_attention)
        {
            auto const layout = torch_ext::TrtllmAttentionWorkspaceManager::buildGenerationLayout(dtype, batch_beam,
                num_tokens, num_heads, head_size, rotary_embedding_dim, num_kv_heads, max_blocks_per_sequence,
                use_sparse_attention);
            nb::dict result;
            result["trtllm_gen_workspace_offset"] = layout.trtllmGenWorkspaceOffset;
            result["cu_seqlens_offset"] = layout.cuSeqlensOffset;
            result["cu_kv_seqlens_offset"] = layout.cuKvSeqlensOffset;
            result["rotary_inv_freq_offset"] = layout.rotaryInvFreqOffset;
            result["tokens_info_offset"] = layout.tokensInfoOffset;
            result["q_buf_offset"] = layout.qBufOffset;
            result["bmm1_scale_offset"] = layout.bmm1ScaleOffset;
            result["bmm2_scale_offset"] = layout.bmm2ScaleOffset;
            result["sparse_attn_cache_offset"] = layout.sparseAttnCacheOffset;
            result["trtllm_gen_workspace_size"] = layout.trtllmGenWorkspaceSize;
            result["cu_seqlens_size"] = layout.cuSeqlensSize;
            result["cu_kv_seqlens_size"] = layout.cuKvSeqlensSize;
            result["rotary_inv_freq_size"] = layout.rotaryInvFreqSize;
            result["tokens_info_size"] = layout.tokensInfoSize;
            result["q_buf_size"] = layout.qBufSize;
            result["bmm1_scale_size"] = layout.bmm1ScaleSize;
            result["bmm2_scale_size"] = layout.bmm2ScaleSize;
            result["sparse_attn_cache_size"] = layout.sparseAttnCacheSize;
            result["total_size"] = layout.totalSize;
            return result;
        },
        nb::arg("dtype"), nb::arg("batch_beam"), nb::arg("num_tokens"), nb::arg("num_heads"), nb::arg("head_size"),
        nb::arg("rotary_embedding_dim"), nb::arg("num_kv_heads"), nb::arg("max_blocks_per_sequence") = 0,
        nb::arg("use_sparse_attention") = false, "Return the C++ trtllm-gen generation workspace layout.");

    m.def("trtllm_gen_context_preprocess", &trtllmGenContextPreprocessBinding, nb::arg("qkv_input"),
        nb::arg("workspace"), nb::arg("sequence_lengths"), nb::arg("context_lengths"),
        nb::arg("kv_cache_block_offsets").none(), nb::arg("host_kv_cache_pool_pointers").none(),
        nb::arg("host_kv_cache_pool_mapping").none(), nb::arg("kv_scale_orig_quant").none(),
        nb::arg("kv_scale_quant_orig").none(), nb::arg("attention_output_orig_quant").none(),
        nb::arg("rotary_inv_freq").none(), nb::arg("rotary_cos_sin").none(), nb::arg("mrope_rotary_cos_sin").none(),
        nb::arg("layer_idx"), nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"),
        nb::arg("tokens_per_block"), nb::arg("mask_type"), nb::arg("kv_cache_quant_mode"),
        nb::arg("max_attention_window_size"), nb::arg("cyclic_attention_window_size"), nb::arg("sink_token_length"),
        nb::arg("num_tokens"), nb::arg("batch_size"), nb::arg("input_seq_length"), nb::arg("max_past_kv_length"),
        nb::arg("rotary_embedding_dim"), nb::arg("rotary_embedding_base"), nb::arg("rotary_embedding_scale_type"),
        nb::arg("rotary_embedding_scale"), nb::arg("rotary_embedding_max_positions"),
        nb::arg("position_embedding_type"), nb::arg("bmm1_scale"), nb::arg("bmm2_scale"),
        nb::arg("attention_chunk_size"), nb::arg("fp8_context_fmha"), nb::arg("paged_context_fmha"),
        nb::arg("is_mla_enable"), nb::arg("multi_processor_count"), nb::arg("total_num_blocks"), nb::arg("kv_factor"),
        nb::arg("need_build_kv_cache_metadata") = true, "Fused nanobind context preprocess for trtllm-gen attention.");

    m.def("trtllm_gen_context_postprocess", &torch_ext::trtllmGenContextPostprocess, nb::arg("qkv_input"),
        nb::arg("workspace"), nb::arg("sequence_lengths"), nb::arg("context_lengths"),
        nb::arg("kv_cache_block_offsets").none(), nb::arg("host_kv_cache_pool_pointers").none(),
        nb::arg("host_kv_cache_pool_mapping").none(), nb::arg("kv_scale_orig_quant").none(),
        nb::arg("kv_scale_quant_orig").none(), nb::arg("attention_output_orig_quant").none(),
        nb::arg("rotary_cos_sin").none(), nb::arg("mrope_rotary_cos_sin").none(), nb::arg("layer_idx"),
        nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"), nb::arg("tokens_per_block"),
        nb::arg("mask_type"), nb::arg("kv_cache_quant_mode"), nb::arg("max_attention_window_size"),
        nb::arg("cyclic_attention_window_size"), nb::arg("sink_token_length"), nb::arg("num_tokens"),
        nb::arg("batch_size"), nb::arg("input_seq_length"), nb::arg("max_past_kv_length"),
        nb::arg("rotary_embedding_dim"), nb::arg("rotary_embedding_base"), nb::arg("rotary_embedding_scale_type"),
        nb::arg("rotary_embedding_scale"), nb::arg("rotary_embedding_max_positions"),
        nb::arg("position_embedding_type"), nb::arg("bmm1_scale"), nb::arg("fp8_context_fmha"),
        nb::arg("paged_context_fmha"), nb::arg("is_mla_enable"), nb::arg("attention_chunk_size"),
        nb::arg("multi_processor_count"), "Fused nanobind context postprocess for trtllm-gen attention.",
        nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "build_trtllm_gen_kv_cache_metadata",
        [](torch::Tensor host_kv_cache_pool_pointers, torch::Tensor host_kv_cache_pool_mapping,
            torch::Tensor kv_cache_block_offsets, int64_t layer_idx, int64_t num_kv_heads, int64_t tokens_per_block,
            int64_t head_dim, int64_t kv_factor, int64_t total_num_blocks, int64_t kv_cache_quant_mode,
            int64_t batch_start, int64_t batch_size, at::ScalarType dtype) -> nb::tuple
        {
            nb::gil_scoped_release release;
            auto kvPool = torch_ext::buildFlashinferTrtllmGenPagedKvCacheBuffers(host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping, layer_idx, num_kv_heads, tokens_per_block, head_dim, kv_factor,
                total_num_blocks, kv_cache_quant_mode, dtype);
            auto const mapping = torch_ext::readKvCachePoolMapping(host_kv_cache_pool_mapping, layer_idx);
            auto blockTables = kv_cache_block_offsets.select(0, mapping.poolIndex).narrow(0, batch_start, batch_size);
            return nb::make_tuple(nb::cast(kvPool), nb::cast(blockTables));
        },
        nb::arg("host_kv_cache_pool_pointers"), nb::arg("host_kv_cache_pool_mapping"),
        nb::arg("kv_cache_block_offsets"), nb::arg("layer_idx"), nb::arg("num_kv_heads"), nb::arg("tokens_per_block"),
        nb::arg("head_dim"), nb::arg("kv_factor"), nb::arg("total_num_blocks"), nb::arg("kv_cache_quant_mode"),
        nb::arg("batch_start"), nb::arg("batch_size"), nb::arg("dtype"),
        "Build flashinfer-style KV cache pool view and slice block tables for a given layer.");

    m.def("trtllm_gen_generation_preprocess", &trtllmGenGenerationPreprocessBinding, nb::arg("qkv_input"),
        nb::arg("workspace"), nb::arg("sequence_lengths"), nb::arg("spec_decoding_generation_lengths").none(),
        nb::arg("spec_decoding_position_offsets").none(), nb::arg("kv_cache_block_offsets").none(),
        nb::arg("host_kv_cache_pool_pointers").none(), nb::arg("host_kv_cache_pool_mapping").none(),
        nb::arg("kv_scale_orig_quant").none(), nb::arg("kv_scale_quant_orig").none(),
        nb::arg("attention_output_orig_quant").none(), nb::arg("rotary_inv_freq").none(),
        nb::arg("rotary_cos_sin").none(), nb::arg("layer_idx"), nb::arg("seq_offset"), nb::arg("num_heads"),
        nb::arg("num_kv_heads"), nb::arg("head_size"), nb::arg("tokens_per_block"), nb::arg("kv_cache_quant_mode"),
        nb::arg("max_attention_window_size"), nb::arg("cyclic_attention_window_size"), nb::arg("sink_token_length"),
        nb::arg("num_tokens"), nb::arg("batch_beam"), nb::arg("input_seq_length"), nb::arg("max_past_kv_length"),
        nb::arg("rotary_embedding_dim"), nb::arg("rotary_embedding_base"), nb::arg("rotary_embedding_scale_type"),
        nb::arg("rotary_embedding_scale"), nb::arg("rotary_embedding_max_positions"),
        nb::arg("position_embedding_type"), nb::arg("bmm1_scale"), nb::arg("bmm2_scale"), nb::arg("fp8_context_fmha"),
        nb::arg("predicted_tokens_per_seq"), nb::arg("attention_chunk_size"), nb::arg("multi_processor_count"),
        nb::arg("total_num_blocks"), nb::arg("kv_factor"), nb::arg("need_build_kv_cache_metadata") = true,
        "Fused nanobind generation preprocess for trtllm-gen attention.");
}
} // namespace tensorrt_llm::nanobind::thop
