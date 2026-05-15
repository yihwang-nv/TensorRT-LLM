#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop


def _data_type_from_torch(dtype: torch.dtype) -> DataType:
    if dtype in (
        DataType.HALF,
        DataType.BF16,
        DataType.FLOAT,
        DataType.FP8,
        DataType.NVFP4,
    ):
        return dtype
    if dtype == torch.float16:
        return DataType.HALF
    if dtype == torch.bfloat16:
        return DataType.BF16
    if dtype == torch.float32:
        return DataType.FLOAT
    if dtype == torch.float8_e4m3fn:
        return DataType.FP8
    if dtype == torch.uint8:
        return DataType.NVFP4
    return DataType.HALF


def _attention_input_type(value: int) -> thop.AttentionInputType:
    if int(value) == 1:
        return thop.AttentionInputType.ContextOnly
    if int(value) == 2:
        return thop.AttentionInputType.GenerationOnly
    return thop.AttentionInputType.Mixed


def _empty_workspace(q: torch.Tensor) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=q.device)


def _optional_tensor_list_value(values, index: int):
    if values is None or len(values) <= index:
        return None
    return values[index]


def _build_spec_dec_args(bool_params, tensor_params):
    if bool_params is None or not bool_params[0]:
        return None
    return thop.TrtllmSpecDecArgs(
        bool(bool_params[1]),
        bool(bool_params[2]),
        _optional_tensor_list_value(tensor_params, 0),
        _optional_tensor_list_value(tensor_params, 1),
        _optional_tensor_list_value(tensor_params, 2),
        _optional_tensor_list_value(tensor_params, 3),
        _optional_tensor_list_value(tensor_params, 4),
        _optional_tensor_list_value(tensor_params, 5),
    )


def _build_helix_args(helix_tensor_params):
    if helix_tensor_params is None or len(helix_tensor_params) == 0:
        return None
    position_offsets = helix_tensor_params[0]
    if position_offsets is None:
        return None
    return thop.TrtllmHelixArgs(position_offsets, helix_tensor_params[1])


def _has_sparse_tensor(tensor: torch.Tensor) -> bool:
    return tensor is not None and tensor.numel() > 0


def _optional_group_presence_changed(
    op_args, kv_cache, mla, sage, sparse, skip_softmax, spec_dec, helix, flash_mla
) -> bool:
    return (
        (op_args.kv_cache is None) != (kv_cache is None)
        or (op_args.mla is None) != (mla is None)
        or (op_args.sage is None) != (sage is None)
        or (op_args.sparse is None) != (sparse is None)
        or (op_args.skip_softmax is None) != (skip_softmax is None)
        or (op_args.spec_dec is None) != (spec_dec is None)
        or (op_args.helix is None) != (helix is None)
        or (op_args.flash_mla is None) != (flash_mla is None)
    )


def _needs_constructor_clear(cached_value, new_value) -> bool:
    return cached_value is not None and new_value is None


def _optional_field_clear_required(
    op_args,
    k,
    v,
    output_sf,
    block_ids_per_seq,
    mrope_rotary_cos_sin,
    mrope_position_deltas,
    kv_cache_block_offsets,
    host_kv_cache_pool_pointers,
    host_kv_cache_pool_mapping,
    cache_indirection,
    compressed_kv_cache_pool_ptr,
    latent_cache,
    q_pe,
    mla_bmm1_scale,
    mla_bmm2_scale,
    quant_q_buffer,
    sparse_kv_indices,
    sparse_kv_offsets,
    sparse_attn_indices,
    sparse_attn_offsets,
    num_sparse_topk,
    sparse_mla_topk_lens,
    skip_softmax_threshold_scale_factor_prefill,
    skip_softmax_threshold_scale_factor_decode,
    skip_softmax_stat,
    spec_decoding_tensor_params,
) -> bool:
    if (
        _needs_constructor_clear(op_args.k, k)
        or _needs_constructor_clear(op_args.v, v)
        or _needs_constructor_clear(op_args.output_sf, output_sf)
        or _needs_constructor_clear(op_args.block_ids_per_seq, block_ids_per_seq)
        or _needs_constructor_clear(op_args.rope.mrope_rotary_cos_sin, mrope_rotary_cos_sin)
        or _needs_constructor_clear(op_args.rope.mrope_position_deltas, mrope_position_deltas)
    ):
        return True

    if op_args.kv_cache is not None:
        cached_kv_cache = op_args.kv_cache
        if (
            _needs_constructor_clear(cached_kv_cache.block_offsets, kv_cache_block_offsets)
            or _needs_constructor_clear(
                cached_kv_cache.host_pool_pointers, host_kv_cache_pool_pointers
            )
            or _needs_constructor_clear(
                cached_kv_cache.host_pool_mapping, host_kv_cache_pool_mapping
            )
            or _needs_constructor_clear(cached_kv_cache.cache_indirection, cache_indirection)
            or _needs_constructor_clear(
                cached_kv_cache.compressed_kv_cache_pool_ptr, compressed_kv_cache_pool_ptr
            )
        ):
            return True

    if op_args.mla is not None:
        cached_mla = op_args.mla
        if (
            _needs_constructor_clear(cached_mla.latent_cache, latent_cache)
            or _needs_constructor_clear(cached_mla.q_pe, q_pe)
            or _needs_constructor_clear(cached_mla.mla_bmm1_scale, mla_bmm1_scale)
            or _needs_constructor_clear(cached_mla.mla_bmm2_scale, mla_bmm2_scale)
            or _needs_constructor_clear(cached_mla.quant_q_buffer, quant_q_buffer)
        ):
            return True

    if op_args.sparse is not None:
        cached_sparse = op_args.sparse
        if (
            _needs_constructor_clear(cached_sparse.sparse_kv_indices, sparse_kv_indices)
            or _needs_constructor_clear(cached_sparse.sparse_kv_offsets, sparse_kv_offsets)
            or _needs_constructor_clear(cached_sparse.sparse_attn_indices, sparse_attn_indices)
            or _needs_constructor_clear(cached_sparse.sparse_attn_offsets, sparse_attn_offsets)
            or _needs_constructor_clear(cached_sparse.num_sparse_topk, num_sparse_topk)
            or _needs_constructor_clear(cached_sparse.sparse_mla_topk_lens, sparse_mla_topk_lens)
        ):
            return True

    if op_args.skip_softmax is not None:
        cached_skip_softmax = op_args.skip_softmax
        if (
            _needs_constructor_clear(
                cached_skip_softmax.threshold_scale_factor_prefill,
                skip_softmax_threshold_scale_factor_prefill,
            )
            or _needs_constructor_clear(
                cached_skip_softmax.threshold_scale_factor_decode,
                skip_softmax_threshold_scale_factor_decode,
            )
            or _needs_constructor_clear(cached_skip_softmax.block_skip_stat, skip_softmax_stat)
        ):
            return True

    if op_args.spec_dec is not None:
        cached_spec_dec = op_args.spec_dec
        if (
            _needs_constructor_clear(
                cached_spec_dec.generation_lengths,
                _optional_tensor_list_value(spec_decoding_tensor_params, 0),
            )
            or _needs_constructor_clear(
                cached_spec_dec.position_offsets,
                _optional_tensor_list_value(spec_decoding_tensor_params, 1),
            )
            or _needs_constructor_clear(
                cached_spec_dec.packed_mask,
                _optional_tensor_list_value(spec_decoding_tensor_params, 2),
            )
            or _needs_constructor_clear(
                cached_spec_dec.bl_tree_mask_offset,
                _optional_tensor_list_value(spec_decoding_tensor_params, 3),
            )
            or _needs_constructor_clear(
                cached_spec_dec.bl_tree_mask,
                _optional_tensor_list_value(spec_decoding_tensor_params, 4),
            )
            or _needs_constructor_clear(
                cached_spec_dec.first_sparse_mask_offset_kv,
                _optional_tensor_list_value(spec_decoding_tensor_params, 5),
            )
        ):
            return True

    return False


def _kv_cache_static_changed(
    cached_kv_cache, tokens_per_block, kv_cache_dtype, q_dtype, is_mla_enable, total_num_blocks
) -> bool:
    if cached_kv_cache is None:
        return False
    return (
        cached_kv_cache.tokens_per_block != (tokens_per_block or 0)
        or cached_kv_cache.dtype
        != _data_type_from_torch(kv_cache_dtype if kv_cache_dtype is not None else q_dtype)
        or cached_kv_cache.kv_factor != (1 if is_mla_enable else 2)
        or cached_kv_cache.total_num_blocks != total_num_blocks
    )


def _mla_static_changed(
    cached_mla,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    rope_append,
) -> bool:
    if cached_mla is None:
        return False
    return (
        cached_mla.q_lora_rank != q_lora_rank
        or cached_mla.kv_lora_rank != kv_lora_rank
        or cached_mla.qk_nope_head_dim != qk_nope_head_dim
        or cached_mla.qk_rope_head_dim != qk_rope_head_dim
        or cached_mla.v_head_dim != v_head_dim
        or cached_mla.rope_append != rope_append
    )


def _assign_if_not_none(obj, attr: str, value):
    if value is not None:
        setattr(obj, attr, value)


def build_trtllm_attention_args(
    q,
    k,
    v,
    output,
    output_sf,
    workspace,
    sequence_length,
    host_past_key_value_lengths,
    host_total_kv_lens,
    context_lengths,
    host_context_lengths,
    host_request_types,
    kv_cache_block_offsets,
    host_kv_cache_pool_pointers,
    host_kv_cache_pool_mapping,
    cache_indirection,
    kv_scale_orig_quant,
    kv_scale_quant_orig,
    out_scale,
    rotary_inv_freq,
    rotary_cos_sin,
    latent_cache,
    q_pe,
    block_ids_per_seq,
    attention_sinks,
    is_fused_qkv,
    update_kv_cache,
    predicted_tokens_per_seq,
    layer_idx,
    num_heads,
    num_kv_heads,
    head_size,
    tokens_per_block,
    max_num_requests,
    max_context_length,
    attention_window_size,
    sink_token_length,
    beam_width,
    mask_type,
    quant_mode,
    q_scaling,
    position_embedding_type,
    rotary_embedding_dim,
    rotary_embedding_base,
    rotary_embedding_scale_type,
    rotary_embedding_scales,
    rotary_embedding_max_position_info,
    use_paged_context_fmha,
    attention_input_type,
    is_mla_enable,
    chunked_prefill_buffer_batch_size,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    rope_append,
    mrope_rotary_cos_sin,
    mrope_position_deltas,
    helix_tensor_params,
    attention_chunk_size,
    softmax_stats_tensor,
    spec_decoding_bool_params,
    spec_decoding_tensor_params,
    sparse_kv_indices,
    sparse_kv_offsets,
    sparse_attn_indices,
    sparse_attn_offsets,
    sparse_attn_indices_block_size,
    num_sparse_topk,
    sparse_mla_topk_lens,
    skip_softmax_threshold_scale_factor_prefill,
    skip_softmax_threshold_scale_factor_decode,
    skip_softmax_stat,
    cu_q_seqlens,
    cu_kv_seqlens,
    fmha_scheduler_counter,
    mla_bmm1_scale,
    mla_bmm2_scale,
    quant_q_buffer,
    flash_mla_tile_scheduler_metadata,
    flash_mla_num_splits,
    sage_attn_num_elts_per_blk_q,
    sage_attn_num_elts_per_blk_k,
    sage_attn_num_elts_per_blk_v,
    sage_attn_qk_int8,
    *,
    num_contexts,
    num_ctx_tokens,
    kv_cache_dtype=None,
    total_num_blocks=0,
    compressed_kv_cache_pool_ptr=None,
    op_args=None,
):
    workspace = workspace if workspace is not None else _empty_workspace(q)
    max_attention_window_size = attention_window_size
    if beam_width != 1 and cache_indirection is not None:
        max_attention_window_size = cache_indirection.size(2)

    quant = thop.TrtllmQuantArgs(kv_scale_orig_quant, kv_scale_quant_orig, out_scale)
    fmha = thop.TrtllmFmhaArgs(
        attention_sinks,
        softmax_stats_tensor,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        chunked_prefill_buffer_batch_size or 1,
    )

    kv_cache = None
    if (
        kv_cache_block_offsets is not None
        or host_kv_cache_pool_pointers is not None
        or host_kv_cache_pool_mapping is not None
    ):
        kv_cache = thop.TrtllmKvCacheArgs(
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            cache_indirection,
            tokens_per_block or 0,
            _data_type_from_torch(kv_cache_dtype if kv_cache_dtype is not None else q.dtype),
            1 if is_mla_enable else 2,
            total_num_blocks,
            compressed_kv_cache_pool_ptr,
        )

    mla = None
    if is_mla_enable:
        mla = thop.TrtllmMlaArgs(
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            rope_append,
            latent_cache,
            q_pe,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
        )

    sage = None
    if (
        sage_attn_num_elts_per_blk_q > 0
        or sage_attn_num_elts_per_blk_k > 0
        or sage_attn_num_elts_per_blk_v > 0
    ):
        sage = thop.TrtllmSageArgs(
            sage_attn_num_elts_per_blk_q,
            sage_attn_num_elts_per_blk_k,
            sage_attn_num_elts_per_blk_v,
            sage_attn_qk_int8,
        )

    sparse = None
    if (
        _has_sparse_tensor(sparse_kv_indices)
        or _has_sparse_tensor(sparse_attn_indices)
        or num_sparse_topk is not None
        or sparse_mla_topk_lens is not None
    ):
        sparse = thop.TrtllmSparseArgs(
            sparse_kv_indices,
            sparse_kv_offsets,
            sparse_attn_indices,
            sparse_attn_offsets,
            sparse_attn_indices_block_size,
            num_sparse_topk,
            sparse_mla_topk_lens,
        )

    skip_softmax = None
    if (
        skip_softmax_threshold_scale_factor_prefill is not None
        or skip_softmax_threshold_scale_factor_decode is not None
        or skip_softmax_stat is not None
    ):
        skip_softmax = thop.TrtllmSkipSoftmaxArgs(
            skip_softmax_threshold_scale_factor_prefill,
            skip_softmax_threshold_scale_factor_decode,
            skip_softmax_stat,
        )

    flash_mla = None
    if flash_mla_tile_scheduler_metadata is not None:
        flash_mla = thop.TrtllmFlashMlaArgs(flash_mla_tile_scheduler_metadata, flash_mla_num_splits)
    spec_dec = _build_spec_dec_args(spec_decoding_bool_params, spec_decoding_tensor_params)
    helix = _build_helix_args(helix_tensor_params)

    if op_args is not None and (
        _optional_group_presence_changed(
            op_args, kv_cache, mla, sage, sparse, skip_softmax, spec_dec, helix, flash_mla
        )
        or _optional_field_clear_required(
            op_args,
            k,
            v,
            output_sf,
            block_ids_per_seq,
            mrope_rotary_cos_sin,
            mrope_position_deltas,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            cache_indirection,
            compressed_kv_cache_pool_ptr,
            latent_cache,
            q_pe,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            sparse_kv_indices,
            sparse_kv_offsets,
            sparse_attn_indices,
            sparse_attn_offsets,
            num_sparse_topk,
            sparse_mla_topk_lens,
            skip_softmax_threshold_scale_factor_prefill,
            skip_softmax_threshold_scale_factor_decode,
            skip_softmax_stat,
            spec_decoding_tensor_params,
        )
        or _kv_cache_static_changed(
            op_args.kv_cache,
            tokens_per_block,
            kv_cache_dtype,
            q.dtype,
            is_mla_enable,
            total_num_blocks,
        )
        or _mla_static_changed(
            op_args.mla,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            rope_append,
        )
    ):
        op_args = None

    if op_args is not None:
        op_args.q = q
        _assign_if_not_none(op_args, "k", k)
        _assign_if_not_none(op_args, "v", v)
        op_args.output = output
        _assign_if_not_none(op_args, "output_sf", output_sf)
        op_args.workspace = workspace
        op_args.layer_idx = layer_idx
        op_args.mask_type = mask_type
        op_args.attention_window_size = attention_window_size
        op_args.max_attention_window_size = max_attention_window_size
        op_args.attention_input_type = _attention_input_type(attention_input_type)
        op_args.is_fused_qkv = is_fused_qkv
        op_args.update_kv_cache = update_kv_cache
        op_args.use_paged_context_fmha = use_paged_context_fmha
        _assign_if_not_none(op_args, "block_ids_per_seq", block_ids_per_seq)
        op_args.sequence_length = sequence_length
        op_args.host_past_key_value_lengths = host_past_key_value_lengths
        op_args.host_total_kv_lens = host_total_kv_lens
        op_args.context_lengths = context_lengths
        op_args.host_context_lengths = host_context_lengths
        op_args.host_request_types = host_request_types
        op_args.num_contexts = num_contexts
        op_args.num_ctx_tokens = num_ctx_tokens
        op_args.max_num_requests = max_num_requests
        op_args.max_context_length = max_context_length
        op_args.beam_width = beam_width
        _assign_if_not_none(op_args.rope, "mrope_rotary_cos_sin", mrope_rotary_cos_sin)
        _assign_if_not_none(op_args.rope, "mrope_position_deltas", mrope_position_deltas)
        op_args.quant = quant
        op_args.fmha = fmha
        if kv_cache is not None:
            cached_kv_cache = op_args.kv_cache
            _assign_if_not_none(cached_kv_cache, "block_offsets", kv_cache_block_offsets)
            _assign_if_not_none(cached_kv_cache, "host_pool_pointers", host_kv_cache_pool_pointers)
            _assign_if_not_none(cached_kv_cache, "host_pool_mapping", host_kv_cache_pool_mapping)
            _assign_if_not_none(cached_kv_cache, "cache_indirection", cache_indirection)
            _assign_if_not_none(
                cached_kv_cache, "compressed_kv_cache_pool_ptr", compressed_kv_cache_pool_ptr
            )
        if mla is not None:
            cached_mla = op_args.mla
            _assign_if_not_none(cached_mla, "latent_cache", latent_cache)
            _assign_if_not_none(cached_mla, "q_pe", q_pe)
            _assign_if_not_none(cached_mla, "mla_bmm1_scale", mla_bmm1_scale)
            _assign_if_not_none(cached_mla, "mla_bmm2_scale", mla_bmm2_scale)
            _assign_if_not_none(cached_mla, "quant_q_buffer", quant_q_buffer)
        if sage is not None:
            cached_sage = op_args.sage
            cached_sage.num_elts_per_blk_q = sage_attn_num_elts_per_blk_q
            cached_sage.num_elts_per_blk_k = sage_attn_num_elts_per_blk_k
            cached_sage.num_elts_per_blk_v = sage_attn_num_elts_per_blk_v
            cached_sage.qk_int8 = sage_attn_qk_int8
        if sparse is not None:
            cached_sparse = op_args.sparse
            _assign_if_not_none(cached_sparse, "sparse_kv_indices", sparse_kv_indices)
            _assign_if_not_none(cached_sparse, "sparse_kv_offsets", sparse_kv_offsets)
            _assign_if_not_none(cached_sparse, "sparse_attn_indices", sparse_attn_indices)
            _assign_if_not_none(cached_sparse, "sparse_attn_offsets", sparse_attn_offsets)
            cached_sparse.sparse_attn_indices_block_size = sparse_attn_indices_block_size
            _assign_if_not_none(cached_sparse, "num_sparse_topk", num_sparse_topk)
            _assign_if_not_none(cached_sparse, "sparse_mla_topk_lens", sparse_mla_topk_lens)
        if skip_softmax is not None:
            cached_skip_softmax = op_args.skip_softmax
            _assign_if_not_none(
                cached_skip_softmax,
                "threshold_scale_factor_prefill",
                skip_softmax_threshold_scale_factor_prefill,
            )
            _assign_if_not_none(
                cached_skip_softmax,
                "threshold_scale_factor_decode",
                skip_softmax_threshold_scale_factor_decode,
            )
            _assign_if_not_none(cached_skip_softmax, "block_skip_stat", skip_softmax_stat)
        if spec_dec is not None:
            cached_spec_dec = op_args.spec_dec
            cached_spec_dec.use_spec_decoding = bool(spec_decoding_bool_params[1])
            cached_spec_dec.is_spec_dec_tree = bool(spec_decoding_bool_params[2])
            _assign_if_not_none(
                cached_spec_dec,
                "generation_lengths",
                _optional_tensor_list_value(spec_decoding_tensor_params, 0),
            )
            _assign_if_not_none(
                cached_spec_dec,
                "position_offsets",
                _optional_tensor_list_value(spec_decoding_tensor_params, 1),
            )
            _assign_if_not_none(
                cached_spec_dec,
                "packed_mask",
                _optional_tensor_list_value(spec_decoding_tensor_params, 2),
            )
            _assign_if_not_none(
                cached_spec_dec,
                "bl_tree_mask_offset",
                _optional_tensor_list_value(spec_decoding_tensor_params, 3),
            )
            _assign_if_not_none(
                cached_spec_dec,
                "bl_tree_mask",
                _optional_tensor_list_value(spec_decoding_tensor_params, 4),
            )
            _assign_if_not_none(
                cached_spec_dec,
                "first_sparse_mask_offset_kv",
                _optional_tensor_list_value(spec_decoding_tensor_params, 5),
            )
        if helix is not None:
            cached_helix = op_args.helix
            cached_helix.position_offsets = helix_tensor_params[0]
            cached_helix.is_inactive_rank = helix_tensor_params[1]
        if flash_mla is not None:
            cached_flash_mla = op_args.flash_mla
            cached_flash_mla.tile_scheduler_metadata = flash_mla_tile_scheduler_metadata
            cached_flash_mla.num_splits = flash_mla_num_splits
        return op_args

    rope = thop.TrtllmRopeArgs(
        position_embedding_type,
        rotary_embedding_dim,
        rotary_embedding_base,
        rotary_embedding_scale_type,
        rotary_embedding_scales,
        rotary_embedding_max_position_info,
        rotary_inv_freq,
        rotary_cos_sin,
        mrope_rotary_cos_sin,
        mrope_position_deltas,
    )

    return thop.TrtllmAttentionArgs(
        q,
        k,
        v,
        output,
        output_sf,
        workspace,
        num_heads,
        num_kv_heads,
        head_size,
        predicted_tokens_per_seq,
        q_scaling,
        quant_mode,
        attention_chunk_size,
        sink_token_length,
        layer_idx,
        mask_type,
        attention_window_size,
        max_attention_window_size,
        _attention_input_type(attention_input_type),
        is_fused_qkv,
        update_kv_cache,
        use_paged_context_fmha,
        block_ids_per_seq,
        sequence_length,
        host_past_key_value_lengths,
        host_total_kv_lens,
        context_lengths,
        host_context_lengths,
        host_request_types,
        num_contexts,
        num_ctx_tokens,
        max_num_requests,
        max_context_length,
        beam_width,
        rope,
        quant,
        fmha,
        kv_cache,
        mla,
        sage,
        sparse,
        skip_softmax,
        spec_dec,
        helix,
        flash_mla,
    )
