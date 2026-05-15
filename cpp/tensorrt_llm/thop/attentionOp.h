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

#pragma once

#include <climits>
#include <optional>
#include <torch/extension.h>
#include <utility>
#include <vector>

#include "tensorrt_llm/common/attentionOp.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

enum class AttentionInputType : int8_t
{
    Mixed,
    ContextOnly,
    GenerationOnly,
};

struct TrtllmRopeArgs
{
    TrtllmRopeArgs(int64_t positionEmbeddingType, int64_t rotaryEmbeddingDim, double rotaryEmbeddingBase,
        int64_t rotaryEmbeddingScaleType, std::vector<double> rotaryEmbeddingScales,
        std::vector<int64_t> rotaryEmbeddingMaxPositionInfo, std::optional<at::Tensor> rotaryInvFreq,
        std::optional<at::Tensor> rotaryCosSin, std::optional<at::Tensor> mropeRotaryCosSin = std::nullopt,
        std::optional<at::Tensor> mropePositionDeltas = std::nullopt)
        : positionEmbeddingType(positionEmbeddingType)
        , rotaryEmbeddingDim(rotaryEmbeddingDim)
        , rotaryEmbeddingBase(rotaryEmbeddingBase)
        , rotaryEmbeddingScaleType(rotaryEmbeddingScaleType)
        , rotaryEmbeddingScales(std::move(rotaryEmbeddingScales))
        , rotaryEmbeddingMaxPositionInfo(std::move(rotaryEmbeddingMaxPositionInfo))
        , rotaryInvFreq(std::move(rotaryInvFreq))
        , rotaryCosSin(std::move(rotaryCosSin))
        , mropeRotaryCosSin(std::move(mropeRotaryCosSin))
        , mropePositionDeltas(std::move(mropePositionDeltas))
    {
    }

    int64_t const positionEmbeddingType;
    int64_t const rotaryEmbeddingDim;
    double const rotaryEmbeddingBase;
    int64_t const rotaryEmbeddingScaleType;
    std::vector<double> const rotaryEmbeddingScales;
    std::vector<int64_t> const rotaryEmbeddingMaxPositionInfo;
    std::optional<at::Tensor> const rotaryInvFreq;
    std::optional<at::Tensor> const rotaryCosSin;
    std::optional<at::Tensor> mropeRotaryCosSin;
    std::optional<at::Tensor> mropePositionDeltas;
};

struct TrtllmQuantArgs
{
    TrtllmQuantArgs(std::optional<at::Tensor> kvScaleOrigQuant = std::nullopt,
        std::optional<at::Tensor> kvScaleQuantOrig = std::nullopt, std::optional<at::Tensor> outScale = std::nullopt)
        : kvScaleOrigQuant(std::move(kvScaleOrigQuant))
        , kvScaleQuantOrig(std::move(kvScaleQuantOrig))
        , outScale(std::move(outScale))
    {
    }

    std::optional<at::Tensor> kvScaleOrigQuant;
    std::optional<at::Tensor> kvScaleQuantOrig;
    std::optional<at::Tensor> outScale;
};

struct TrtllmFmhaArgs
{
    TrtllmFmhaArgs(std::optional<at::Tensor> attentionSinks = std::nullopt,
        std::optional<at::Tensor> softmaxStatsTensor = std::nullopt,
        std::optional<at::Tensor> cuQSeqlens = std::nullopt, std::optional<at::Tensor> cuKvSeqlens = std::nullopt,
        std::optional<at::Tensor> fmhaSchedulerCounter = std::nullopt, int64_t chunkedPrefillBufferBatchSize = 1)
        : attentionSinks(std::move(attentionSinks))
        , softmaxStatsTensor(std::move(softmaxStatsTensor))
        , cuQSeqlens(std::move(cuQSeqlens))
        , cuKvSeqlens(std::move(cuKvSeqlens))
        , fmhaSchedulerCounter(std::move(fmhaSchedulerCounter))
        , chunkedPrefillBufferBatchSize(chunkedPrefillBufferBatchSize)
    {
    }

    std::optional<at::Tensor> attentionSinks;
    std::optional<at::Tensor> softmaxStatsTensor;
    std::optional<at::Tensor> cuQSeqlens;
    std::optional<at::Tensor> cuKvSeqlens;
    std::optional<at::Tensor> fmhaSchedulerCounter;
    int64_t chunkedPrefillBufferBatchSize{1};
};

struct TrtllmKvCacheArgs
{
    TrtllmKvCacheArgs(std::optional<at::Tensor> blockOffsets, std::optional<at::Tensor> hostPoolPointers,
        std::optional<at::Tensor> hostPoolMapping, std::optional<at::Tensor> cacheIndirection, int64_t tokensPerBlock,
        nvinfer1::DataType dtype, int64_t kvFactor, int64_t totalNumBlocks,
        std::optional<int64_t> compressedKvCachePoolPtr = std::nullopt)
        : blockOffsets(std::move(blockOffsets))
        , hostPoolPointers(std::move(hostPoolPointers))
        , hostPoolMapping(std::move(hostPoolMapping))
        , cacheIndirection(std::move(cacheIndirection))
        , tokensPerBlock(tokensPerBlock)
        , dtype(dtype)
        , kvFactor(kvFactor)
        , totalNumBlocks(totalNumBlocks)
        , compressedKvCachePoolPtr(compressedKvCachePoolPtr)
    {
    }

    std::optional<at::Tensor> blockOffsets;
    std::optional<at::Tensor> hostPoolPointers;
    std::optional<at::Tensor> hostPoolMapping;
    std::optional<at::Tensor> cacheIndirection;
    int64_t const tokensPerBlock;
    nvinfer1::DataType const dtype;
    int64_t const kvFactor;
    int64_t const totalNumBlocks;
    std::optional<int64_t> compressedKvCachePoolPtr;
};

struct TrtllmMlaArgs
{
    TrtllmMlaArgs(std::optional<int64_t> qLoraRank, int64_t kvLoraRank, int64_t qkNopeHeadDim, int64_t qkRopeHeadDim,
        int64_t vHeadDim, bool ropeAppend, std::optional<at::Tensor> latentCache = std::nullopt,
        std::optional<at::Tensor> qPe = std::nullopt, std::optional<at::Tensor> mlaBmm1Scale = std::nullopt,
        std::optional<at::Tensor> mlaBmm2Scale = std::nullopt, std::optional<at::Tensor> quantQBuffer = std::nullopt)
        : qLoraRank(qLoraRank)
        , kvLoraRank(kvLoraRank)
        , qkNopeHeadDim(qkNopeHeadDim)
        , qkRopeHeadDim(qkRopeHeadDim)
        , vHeadDim(vHeadDim)
        , ropeAppend(ropeAppend)
        , latentCache(std::move(latentCache))
        , qPe(std::move(qPe))
        , mlaBmm1Scale(std::move(mlaBmm1Scale))
        , mlaBmm2Scale(std::move(mlaBmm2Scale))
        , quantQBuffer(std::move(quantQBuffer))
    {
    }

    std::optional<int64_t> const qLoraRank;
    int64_t const kvLoraRank;
    int64_t const qkNopeHeadDim;
    int64_t const qkRopeHeadDim;
    int64_t const vHeadDim;
    bool const ropeAppend;
    std::optional<at::Tensor> latentCache;
    std::optional<at::Tensor> qPe;
    std::optional<at::Tensor> mlaBmm1Scale;
    std::optional<at::Tensor> mlaBmm2Scale;
    std::optional<at::Tensor> quantQBuffer;
};

struct TrtllmSageArgs
{
    TrtllmSageArgs(int64_t numEltsPerBlkQ, int64_t numEltsPerBlkK, int64_t numEltsPerBlkV, bool qkInt8)
        : numEltsPerBlkQ(numEltsPerBlkQ)
        , numEltsPerBlkK(numEltsPerBlkK)
        , numEltsPerBlkV(numEltsPerBlkV)
        , qkInt8(qkInt8)
    {
    }

    int64_t numEltsPerBlkQ;
    int64_t numEltsPerBlkK;
    int64_t numEltsPerBlkV;
    bool qkInt8;
};

struct TrtllmSparseArgs
{
    TrtllmSparseArgs(std::optional<at::Tensor> sparseKvIndices = std::nullopt,
        std::optional<at::Tensor> sparseKvOffsets = std::nullopt,
        std::optional<at::Tensor> sparseAttnIndices = std::nullopt,
        std::optional<at::Tensor> sparseAttnOffsets = std::nullopt, int64_t sparseAttnIndicesBlockSize = 0,
        std::optional<int64_t> numSparseTopk = std::nullopt, std::optional<at::Tensor> sparseMlaTopkLens = std::nullopt)
        : sparseKvIndices(std::move(sparseKvIndices))
        , sparseKvOffsets(std::move(sparseKvOffsets))
        , sparseAttnIndices(std::move(sparseAttnIndices))
        , sparseAttnOffsets(std::move(sparseAttnOffsets))
        , sparseAttnIndicesBlockSize(sparseAttnIndicesBlockSize)
        , numSparseTopk(numSparseTopk)
        , sparseMlaTopkLens(std::move(sparseMlaTopkLens))
    {
    }

    std::optional<at::Tensor> sparseKvIndices;
    std::optional<at::Tensor> sparseKvOffsets;
    std::optional<at::Tensor> sparseAttnIndices;
    std::optional<at::Tensor> sparseAttnOffsets;
    int64_t sparseAttnIndicesBlockSize;
    std::optional<int64_t> numSparseTopk;
    std::optional<at::Tensor> sparseMlaTopkLens;
};

struct TrtllmSkipSoftmaxArgs
{
    TrtllmSkipSoftmaxArgs(std::optional<double> thresholdScaleFactorPrefill = std::nullopt,
        std::optional<double> thresholdScaleFactorDecode = std::nullopt,
        std::optional<at::Tensor> blockSkipStat = std::nullopt)
        : thresholdScaleFactorPrefill(thresholdScaleFactorPrefill)
        , thresholdScaleFactorDecode(thresholdScaleFactorDecode)
        , blockSkipStat(std::move(blockSkipStat))
    {
    }

    std::optional<double> thresholdScaleFactorPrefill;
    std::optional<double> thresholdScaleFactorDecode;
    std::optional<at::Tensor> blockSkipStat;
};

struct TrtllmSpecDecArgs
{
    TrtllmSpecDecArgs(bool useSpecDecoding, bool isSpecDecTree,
        std::optional<at::Tensor> generationLengths = std::nullopt,
        std::optional<at::Tensor> positionOffsets = std::nullopt, std::optional<at::Tensor> packedMask = std::nullopt,
        std::optional<at::Tensor> blTreeMaskOffset = std::nullopt, std::optional<at::Tensor> blTreeMask = std::nullopt,
        std::optional<at::Tensor> firstSparseMaskOffsetKv = std::nullopt)
        : useSpecDecoding(useSpecDecoding)
        , isSpecDecTree(isSpecDecTree)
        , generationLengths(std::move(generationLengths))
        , positionOffsets(std::move(positionOffsets))
        , packedMask(std::move(packedMask))
        , blTreeMaskOffset(std::move(blTreeMaskOffset))
        , blTreeMask(std::move(blTreeMask))
        , firstSparseMaskOffsetKv(std::move(firstSparseMaskOffsetKv))
    {
    }

    bool useSpecDecoding;
    bool isSpecDecTree;
    std::optional<at::Tensor> generationLengths;
    std::optional<at::Tensor> positionOffsets;
    std::optional<at::Tensor> packedMask;
    std::optional<at::Tensor> blTreeMaskOffset;
    std::optional<at::Tensor> blTreeMask;
    std::optional<at::Tensor> firstSparseMaskOffsetKv;
};

struct TrtllmHelixArgs
{
    TrtllmHelixArgs(at::Tensor positionOffsets, at::Tensor isInactiveRank)
        : positionOffsets(std::move(positionOffsets))
        , isInactiveRank(std::move(isInactiveRank))
    {
    }

    at::Tensor positionOffsets;
    at::Tensor isInactiveRank;
};

struct TrtllmFlashMlaArgs
{
    TrtllmFlashMlaArgs(at::Tensor tileSchedulerMetadata, at::Tensor numSplits)
        : tileSchedulerMetadata(std::move(tileSchedulerMetadata))
        , numSplits(std::move(numSplits))
    {
    }

    at::Tensor tileSchedulerMetadata;
    at::Tensor numSplits;
};

struct TrtllmAttentionArgs
{
    TrtllmAttentionArgs(at::Tensor q, std::optional<at::Tensor> k, std::optional<at::Tensor> v, at::Tensor output,
        std::optional<at::Tensor> outputSf, at::Tensor workspace, int64_t numHeads, int64_t numKvHeads,
        int64_t headSize, int64_t predictedTokensPerSeq, double qScaling, int64_t quantMode,
        std::optional<int64_t> attentionChunkSize, int64_t sinkTokenLength, int64_t layerIdx, int64_t maskType,
        int64_t attentionWindowSize, int64_t maxAttentionWindowSize, AttentionInputType attentionInputType,
        bool isFusedQkv, bool updateKvCache, bool usePagedContextFmha, std::optional<at::Tensor> blockIdsPerSeq,
        at::Tensor sequenceLength, at::Tensor hostPastKeyValueLengths, at::Tensor hostTotalKvLens,
        at::Tensor contextLengths, at::Tensor hostContextLengths, at::Tensor hostRequestTypes, int64_t numContexts,
        int64_t numCtxTokens, int64_t maxNumRequests, int64_t maxContextLength, int64_t beamWidth, TrtllmRopeArgs rope,
        TrtllmQuantArgs quant, TrtllmFmhaArgs fmha, std::optional<TrtllmKvCacheArgs> kvCache = std::nullopt,
        std::optional<TrtllmMlaArgs> mla = std::nullopt, std::optional<TrtllmSageArgs> sage = std::nullopt,
        std::optional<TrtllmSparseArgs> sparse = std::nullopt,
        std::optional<TrtllmSkipSoftmaxArgs> skipSoftmax = std::nullopt,
        std::optional<TrtllmSpecDecArgs> specDec = std::nullopt, std::optional<TrtllmHelixArgs> helix = std::nullopt,
        std::optional<TrtllmFlashMlaArgs> flashMla = std::nullopt)
        : q(std::move(q))
        , k(std::move(k))
        , v(std::move(v))
        , output(std::move(output))
        , outputSf(std::move(outputSf))
        , workspace(std::move(workspace))
        , numHeads(numHeads)
        , numKvHeads(numKvHeads)
        , headSize(headSize)
        , predictedTokensPerSeq(predictedTokensPerSeq)
        , qScaling(qScaling)
        , quantMode(quantMode)
        , attentionChunkSize(attentionChunkSize)
        , sinkTokenLength(sinkTokenLength)
        , layerIdx(layerIdx)
        , maskType(maskType)
        , attentionWindowSize(attentionWindowSize)
        , maxAttentionWindowSize(maxAttentionWindowSize)
        , attentionInputType(attentionInputType)
        , isFusedQkv(isFusedQkv)
        , updateKvCache(updateKvCache)
        , usePagedContextFmha(usePagedContextFmha)
        , blockIdsPerSeq(std::move(blockIdsPerSeq))
        , sequenceLength(std::move(sequenceLength))
        , hostPastKeyValueLengths(std::move(hostPastKeyValueLengths))
        , hostTotalKvLens(std::move(hostTotalKvLens))
        , contextLengths(std::move(contextLengths))
        , hostContextLengths(std::move(hostContextLengths))
        , hostRequestTypes(std::move(hostRequestTypes))
        , numContexts(numContexts)
        , numCtxTokens(numCtxTokens)
        , maxNumRequests(maxNumRequests)
        , maxContextLength(maxContextLength)
        , beamWidth(beamWidth)
        , rope(std::move(rope))
        , quant(std::move(quant))
        , fmha(std::move(fmha))
        , kvCache(std::move(kvCache))
        , mla(std::move(mla))
        , sage(std::move(sage))
        , sparse(std::move(sparse))
        , skipSoftmax(std::move(skipSoftmax))
        , specDec(std::move(specDec))
        , helix(std::move(helix))
        , flashMla(std::move(flashMla))
    {
    }

    at::Tensor q;
    std::optional<at::Tensor> k;
    std::optional<at::Tensor> v;
    at::Tensor output;
    std::optional<at::Tensor> outputSf;
    at::Tensor workspace;
    int64_t const numHeads;
    int64_t const numKvHeads;
    int64_t const headSize;
    int64_t const predictedTokensPerSeq;
    double const qScaling;
    int64_t const quantMode;
    std::optional<int64_t> const attentionChunkSize;
    int64_t const sinkTokenLength;
    int64_t layerIdx;
    int64_t maskType;
    int64_t attentionWindowSize;
    int64_t maxAttentionWindowSize;
    AttentionInputType attentionInputType;
    bool isFusedQkv;
    bool updateKvCache;
    bool usePagedContextFmha;
    std::optional<at::Tensor> blockIdsPerSeq;
    at::Tensor sequenceLength;
    at::Tensor hostPastKeyValueLengths;
    at::Tensor hostTotalKvLens;
    at::Tensor contextLengths;
    at::Tensor hostContextLengths;
    at::Tensor hostRequestTypes;
    int64_t numContexts;
    int64_t numCtxTokens;
    int64_t maxNumRequests;
    int64_t maxContextLength;
    int64_t beamWidth;
    TrtllmRopeArgs const rope;
    TrtllmQuantArgs quant;
    TrtllmFmhaArgs fmha;
    std::optional<TrtllmKvCacheArgs> kvCache;
    std::optional<TrtllmMlaArgs> mla;
    std::optional<TrtllmSageArgs> sage;
    std::optional<TrtllmSparseArgs> sparse;
    std::optional<TrtllmSkipSoftmaxArgs> skipSoftmax;
    std::optional<TrtllmSpecDecArgs> specDec;
    std::optional<TrtllmHelixArgs> helix;
    std::optional<TrtllmFlashMlaArgs> flashMla;
};

/**
 * @brief Attention operation for TensorRT-LLM
 *
 * This function performs multi-head attention computation in-place, supporting both
 * context and generation phases with various optimization features including:
 * - Fused QKV processing
 * - KV cache management
 * - Multiple position embedding types (RoPE, ALiBi, etc.)
 * - Quantization support (FP8, FP4, etc.)
 * - Multi-layer attention (MLA)
 * - Speculative decoding
 */
void attention(TrtllmAttentionArgs const& args);

int64_t getAttentionWorkspaceSize(TrtllmAttentionArgs const& args);

struct KvCachePoolPointers
{
    void* primaryPoolPtr{nullptr};
    void* secondaryPoolPtr{nullptr};
    void* primaryBlockScalePoolPtr{nullptr};
    void* secondaryBlockScalePoolPtr{nullptr};
};

struct KvCachePoolMapping
{
    int32_t poolIndex{0};
    int32_t layerIdxInCachePool{0};
};

KvCachePoolMapping readKvCachePoolMapping(at::Tensor const& hostKvCachePoolMapping, int64_t layerIdx);

KvCachePoolPointers buildKvCachePoolPointers(at::Tensor const& hostKvCachePoolPointers, int32_t poolIndex,
    int64_t intraPoolOffset, int64_t blockSize, int32_t layerIdxInCachePool, int32_t kvFactor, bool isFp4KvCache);

common::op::KvCacheBuffers<kernels::KVBlockArray> buildPagedKvCacheBuffers(
    std::optional<torch::Tensor> const& kv_cache_block_offsets,
    std::optional<torch::Tensor> const& host_kv_cache_pool_pointers,
    std::optional<torch::Tensor> const& host_kv_cache_pool_mapping, common::QuantMode quantMode, int64_t layer_idx,
    int64_t batch_size, int64_t tokens_per_block, int64_t kv_head_num, int64_t size_per_head,
    int64_t cyclic_attention_window_size, int64_t max_attention_window_size, int64_t sink_token_length,
    int64_t beam_width, int64_t seq_offset, bool is_mla_enable, size_t elem_size);

at::Tensor buildFlashinferTrtllmGenPagedKvCacheBuffers(at::Tensor host_kv_cache_pool_pointers,
    at::Tensor host_kv_cache_pool_mapping, int64_t layer_idx, int64_t num_kv_heads, int64_t tokens_per_block,
    int64_t head_dim, int64_t kv_factor, int64_t total_num_blocks, int64_t kv_cache_quant_mode, at::ScalarType dtype);

// Layout manager for the thop attention workspace slices used by trtllm-gen.
// Context follows AttentionOp::getWorkspaceSizeForContext() ordering. Generation
// follows the XQA workspace ordering used by AttentionOp generation.
struct TrtllmGenContextWorkspaceLayout
{
    int64_t trtllmGenWorkspaceOffset{};
    int64_t cuQSeqlensOffset{};
    int64_t cuKvSeqlensOffset{};
    int64_t cuMaskRowsOffset{};
    int64_t rotaryInvFreqOffset{};
    int64_t qBufOffset{};
    int64_t tokensInfoOffset{};
    int64_t fmhaTileCounterOffset{};
    int64_t fmhaBmm1ScaleOffset{};
    int64_t fmhaBmm2ScaleOffset{};
    int64_t trtllmGenWorkspaceSize{};
    int64_t cuSeqlensSize{};
    int64_t rotaryInvFreqSize{};
    int64_t qBufSize{};
    int64_t tokensInfoSize{};
    int64_t fmhaTileCounterSize{};
    int64_t fmhaBmm1ScaleSize{};
    int64_t fmhaBmm2ScaleSize{};
    int64_t totalSize{};
    at::ScalarType qBufScalarType{};
};

struct TrtllmGenGenerationWorkspaceLayout
{
    int64_t trtllmGenWorkspaceOffset{};
    int64_t cuSeqlensOffset{};
    int64_t cuKvSeqlensOffset{};
    int64_t rotaryInvFreqOffset{};
    int64_t tokensInfoOffset{};
    int64_t qBufOffset{};
    int64_t bmm1ScaleOffset{};
    int64_t bmm2ScaleOffset{};
    int64_t sparseAttnCacheOffset{};
    int64_t trtllmGenWorkspaceSize{};
    int64_t cuSeqlensSize{};
    int64_t cuKvSeqlensSize{};
    int64_t rotaryInvFreqSize{};
    int64_t tokensInfoSize{};
    int64_t qBufSize{};
    int64_t bmm1ScaleSize{};
    int64_t bmm2ScaleSize{};
    int64_t sparseAttnCacheSize{};
    int64_t totalSize{};
    at::ScalarType qBufScalarType{};
};

struct TrtllmGenContextWorkspaceViews
{
    at::Tensor trtllmGenWorkspace;
    at::Tensor cuQSeqlens;
    at::Tensor cuKvSeqlens;
    at::Tensor cuMaskRows;
    std::optional<at::Tensor> rotaryInvFreqBuf;
    std::optional<at::Tensor> qBuf;
    at::Tensor tokensInfo;
    at::Tensor fmhaTileCounter;
    std::optional<at::Tensor> fmhaBmm1Scale;
    std::optional<at::Tensor> fmhaBmm2Scale;
};

struct TrtllmGenGenerationWorkspaceViews
{
    at::Tensor trtllmGenWorkspace;
    at::Tensor cuSeqlens;
    at::Tensor cuKvSeqlens;
    std::optional<at::Tensor> rotaryInvFreqBuf;
    at::Tensor tokensInfo;
    at::Tensor qBuf;
    at::Tensor bmm1Scale;
    at::Tensor bmm2Scale;
    std::optional<at::Tensor> sparseAttnCache;
};

class TrtllmAttentionWorkspaceManager
{
public:
    static constexpr int64_t kWorkspaceAlignment = 256;
    static constexpr int64_t kTrtllmGenWorkspaceSize = CUBLAS_WORKSPACE_SIZE;

    static TrtllmGenContextWorkspaceLayout buildContextLayout(at::ScalarType qDtype, int64_t batchSize,
        int64_t numTokens, int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, bool separateQKvInput,
        bool fp8ContextFmha);

    static TrtllmGenGenerationWorkspaceLayout buildGenerationLayout(at::ScalarType qDtype, int64_t batchBeam,
        int64_t numTokens, int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, int64_t numKvHeads,
        int64_t maxBlocksPerSequence, bool useSparseAttention);

    static int64_t getContextWorkspaceSize(at::ScalarType qDtype, int64_t batchSize, int64_t numTokens,
        int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, bool separateQKvInput, bool fp8ContextFmha);

    //! numKvHeads and maxBlocksPerSequence affect the size only when sparse attention is enabled.
    static int64_t getGenerationWorkspaceSize(at::ScalarType qDtype, int64_t batchBeam, int64_t numTokens,
        int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, int64_t numKvHeads,
        int64_t maxBlocksPerSequence, bool useSparseAttention);

    static TrtllmGenContextWorkspaceViews materializeContextWorkspace(
        at::Tensor const& workspace, TrtllmGenContextWorkspaceLayout const& layout);

    static TrtllmGenContextWorkspaceViews materializeContextWorkspace(at::Tensor const& workspace,
        at::ScalarType qDtype, int64_t batchSize, int64_t numTokens, int64_t numHeads, int64_t headSize,
        int64_t rotaryEmbeddingDim, bool fp8ContextFmha);

    static TrtllmGenGenerationWorkspaceViews materializeGenerationWorkspace(
        at::Tensor const& workspace, TrtllmGenGenerationWorkspaceLayout const& layout);

    static TrtllmGenGenerationWorkspaceViews materializeGenerationWorkspace(at::Tensor const& workspace,
        at::ScalarType qDtype, int64_t batchBeam, int64_t numTokens, int64_t numHeads, int64_t headSize,
        int64_t rotaryEmbeddingDim, int64_t numKvHeads);

private:
    static std::optional<at::Tensor> makeWorkspaceView(
        at::Tensor const& workspace, int64_t offset, int64_t sizeBytes, at::ScalarType scalarType);
};

} // namespace torch_ext

/**
 * @brief Compute FlashMLA tile-scheduler metadata in-place.
 *
 * Call once per forward pass before the attention layers to pre-compute
 * get_mla_metadata and store the results in the provided tensors. Pass
 * these tensors to the attention op so all layers reuse the same metadata.
 */
void computeFlashMlaMetadata(torch::Tensor seqlens_k, torch::Tensor tile_scheduler_metadata, torch::Tensor num_splits,
    int64_t batch_size, int64_t s_q, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_size_v);

TRTLLM_NAMESPACE_END
