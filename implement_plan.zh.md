<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
-->

# TrtllmAttentionArgs 参数设计

范围：本文档定义 args 分类、struct shape，以及新的 `thop.attention` /
`Runner` interface shape。Staged migration、direct-caller 更新、validation
matrix 和 benchmark plan 应在该设计确认后再写。

## 目标

`thop.attention` 和 TRTLLM-Gen 应消费同一套 attention 参数契约。在改调用点之前，
先把 `TrtllmAttentionArgs` 的参数分类定清楚：结构要简单、清晰、可扩展，并且构造开销低。

设计规则：

- 没有实际语义增益的字段保持顶层 flat，不为了分类而嵌套。
- 只有存在明确 subsystem 边界或 feature activation 语义时才建 substruct。
- layer-static 字段使用 `const` member，让类型表达“构造时确定，之后不可改”。
- per-call 字段保持 mutable，方便 builder 和定向测试。

## 参数分类

最终 top-level shape 应该基本保持 flat：

```text
TrtllmAttentionArgs
├── I/O fields                         per-call, flat
├── Layer static fields                const, flat
├── Layer per-call fields              mutable, flat
├── Batch fields                       per-call, flat
├── rope: TrtllmRopeArgs               RoPE subsystem
├── quant: TrtllmQuantArgs             resolved quant scales
├── fmha: TrtllmFmhaArgs               FMHA auxiliary inputs/outputs
└── Optional feature groups            presence means feature active
    ├── kvCache
    ├── mla
    ├── sage
    ├── sparse
    ├── skipSoftmax
    ├── specDec
    ├── helix
    └── flashMla
```

暂时不要把 `io`、`batch`、`layer` 拆成 substruct。这些分组大多只是复述已有参数类别，
会让调用点更啰嗦，但不带来更强的 invariant。

## Struct Shape

### RoPE

RoPE 有足够强的局部性，适合作为 substruct。Static RoPE fields 是 `const`；
mRoPE tensors 是 per-call。

```cpp
struct TrtllmRopeArgs
{
    // Static: set from layer cache.
    const int64_t positionEmbeddingType;
    const int64_t rotaryEmbeddingDim;
    const double rotaryEmbeddingBase;
    const int64_t rotaryEmbeddingScaleType;
    const std::vector<double> rotaryEmbeddingScales;
    const std::vector<int64_t> rotaryEmbeddingMaxPositionInfo;
    const std::optional<at::Tensor> rotaryInvFreq;
    const std::optional<at::Tensor> rotaryCosSin;

    // Per-call: unpacked from AttentionForwardArgs::mrope_config.
    std::optional<at::Tensor> mropeRotaryCosSin;
    std::optional<at::Tensor> mropePositionDeltas;
};
```

### Quant

这些都是 per-call resolved values。Builder 优先使用 forward override，否则使用 layer
default。

```cpp
struct TrtllmQuantArgs
{
    std::optional<at::Tensor> kvScaleOrigQuant;
    std::optional<at::Tensor> kvScaleQuantOrig;
    std::optional<at::Tensor> outScale;
};
```

### FMHA

`softmaxStatsTensor` 放在这里。它是 FMHA softmax-stats output buffer，用于 Helix 和
MLA merge 路径，不是 skip-softmax feature flag。

```cpp
struct TrtllmFmhaArgs
{
    std::optional<at::Tensor> attentionSinks;
    std::optional<at::Tensor> softmaxStatsTensor;
    std::optional<at::Tensor> cuQSeqlens;
    std::optional<at::Tensor> cuKvSeqlens;
    std::optional<at::Tensor> fmhaSchedulerCounter;
    int64_t chunkedPrefillBufferBatchSize{1};
};
```

### Optional Feature Groups

Feature groups 都是 optional。`std::nullopt` 表示 feature inactive。

```cpp
struct TrtllmKvCacheArgs
{
    std::optional<at::Tensor> blockOffsets;
    std::optional<at::Tensor> hostPoolPointers;
    std::optional<at::Tensor> hostPoolMapping;
    std::optional<at::Tensor> cacheIndirection;
    const int64_t tokensPerBlock;
    const DataType dtype;
    const int64_t kvFactor;
    const int64_t totalNumBlocks;
    std::optional<int64_t> compressedKvCachePoolPtr;
};

struct TrtllmMlaArgs
{
    // Static MLA geometry.
    const std::optional<int64_t> qLoraRank;
    const int64_t kvLoraRank;
    const int64_t qkNopeHeadDim;
    const int64_t qkRopeHeadDim;
    const int64_t vHeadDim;
    const bool ropeAppend;

    // Per-call MLA tensors. Keep optional until every active MLA path is proven
    // to require them.
    std::optional<at::Tensor> latentCache;
    std::optional<at::Tensor> qPe;
    std::optional<at::Tensor> mlaBmm1Scale;
    std::optional<at::Tensor> mlaBmm2Scale;
    std::optional<at::Tensor> quantQBuffer;
};

struct TrtllmSageArgs
{
    int64_t numEltsPerBlkQ;
    int64_t numEltsPerBlkK;
    int64_t numEltsPerBlkV;
    bool qkInt8;
};

struct TrtllmSparseArgs
{
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
    std::optional<double> thresholdScaleFactorPrefill;
    std::optional<double> thresholdScaleFactorDecode;
    std::optional<at::Tensor> blockSkipStat;
};

struct TrtllmSpecDecArgs
{
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
    at::Tensor positionOffsets;
    at::Tensor isInactiveRank;
};

struct TrtllmFlashMlaArgs
{
    at::Tensor tileSchedulerMetadata;
    at::Tensor numSplits;
};
```

### Top-Level Args

```cpp
struct TrtllmAttentionArgs
{
    // I/O: per-call.
    at::Tensor q;
    std::optional<at::Tensor> k;
    std::optional<at::Tensor> v;
    at::Tensor output;
    std::optional<at::Tensor> outputSf;
    // Required scratch buffer. Python selects it before calling the workspace
    // sizing query and sizes it before calling thop.attention(args). C++ does
    // not allocate or resize it.
    at::Tensor workspace;

    // Layer static: set at construction.
    const int64_t numHeads;
    const int64_t numKvHeads;
    const int64_t headSize;
    const int64_t predictedTokensPerSeq;
    const double qScaling;
    const int64_t quantMode;
    const std::optional<int64_t> attentionChunkSize;
    // Currently always 0; static slot reserved for streaming-LLM support.
    const int64_t sinkTokenLength;

    // Layer per-call.
    // layerIdx is per-call because KV cache manager can remap global layer id
    // to local cache-pool layer id.
    int64_t layerIdx;
    int64_t maskType;
    int64_t attentionWindowSize;
    int64_t maxAttentionWindowSize;
    AttentionInputType attentionInputType;
    bool isFusedQkv;
    bool updateKvCache;
    bool usePagedContextFmha;
    std::optional<at::Tensor> blockIdsPerSeq;

    // Batch state: per-call.
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

    // Always-present subsystem substructs.
    TrtllmRopeArgs rope;
    TrtllmQuantArgs quant;
    TrtllmFmhaArgs fmha;

    // Optional features.
    std::optional<TrtllmKvCacheArgs> kvCache;
    std::optional<TrtllmMlaArgs> mla;
    std::optional<TrtllmSageArgs> sage;
    std::optional<TrtllmSparseArgs> sparse;
    std::optional<TrtllmSkipSoftmaxArgs> skipSoftmax;
    std::optional<TrtllmSpecDecArgs> specDec;
    std::optional<TrtllmHelixArgs> helix;
    std::optional<TrtllmFlashMlaArgs> flashMla;
};
```

## Const 与构造

包含 static state 的 struct 使用 `const` fields 后，whole-object assignment 会被删除。
这是可以接受的，前提是 builder 使用 one-shot construction：

```cpp
TrtllmAttentionArgs args{
    .q = q,
    .k = k,
    .v = v,
    .output = output,
    .outputSf = outputSf,
    .workspace = workspace,
    .numHeads = numHeads,
    .numKvHeads = numKvHeads,
    .headSize = headSize,
    .predictedTokensPerSeq = predictedTokensPerSeq,
    .qScaling = qScaling,
    .quantMode = quantMode,
    .attentionChunkSize = attentionChunkSize,
    .sinkTokenLength = sinkTokenLength,
    .layerIdx = layerIdx,
    .maskType = maskType,
    .attentionWindowSize = attentionWindowSize,
    .maxAttentionWindowSize = maxAttentionWindowSize,
    .attentionInputType = attentionInputType,
    .isFusedQkv = isFusedQkv,
    .updateKvCache = updateKvCache,
    .usePagedContextFmha = usePagedContextFmha,
    .blockIdsPerSeq = blockIdsPerSeq,
    .sequenceLength = sequenceLength,
    .hostPastKeyValueLengths = hostPastKeyValueLengths,
    .hostTotalKvLens = hostTotalKvLens,
    .contextLengths = contextLengths,
    .hostContextLengths = hostContextLengths,
    .hostRequestTypes = hostRequestTypes,
    .numContexts = numContexts,
    .numCtxTokens = numCtxTokens,
    .maxNumRequests = maxNumRequests,
    .maxContextLength = maxContextLength,
    .beamWidth = beamWidth,
    .rope = TrtllmRopeArgs{
        .positionEmbeddingType = positionEmbeddingType,
        .rotaryEmbeddingDim = rotaryEmbeddingDim,
        .rotaryEmbeddingBase = rotaryEmbeddingBase,
        .rotaryEmbeddingScaleType = rotaryEmbeddingScaleType,
        .rotaryEmbeddingScales = rotaryEmbeddingScales,
        .rotaryEmbeddingMaxPositionInfo = rotaryEmbeddingMaxPositionInfo,
        .rotaryInvFreq = rotaryInvFreq,
        .rotaryCosSin = rotaryCosSin,
        .mropeRotaryCosSin = mropeRotaryCosSin,
        .mropePositionDeltas = mropePositionDeltas,
    },
    .quant = TrtllmQuantArgs{...},
    .fmha = TrtllmFmhaArgs{...},
    .kvCache = kvCache,
    .mla = mla,
    .sage = sage,
    .sparse = sparse,
    .skipSoftmax = skipSoftmax,
    .specDec = specDec,
    .helix = helix,
    .flashMla = flashMla,
};
```

Nanobind binding 中 static fields 用 `def_ro`，per-call fields 用 `def_rw`。对于包含
`const` member 的 substruct，优先提供 constructor 和 static field read-only exposure，
不要依赖 whole-substruct replacement。

包含 `const` member 的 always-present substruct，例如 `rope`，必须在 top-level
construction expression 中初始化，不能事后赋值。Optional feature group 不同：
`args.mla = TrtllmMlaArgs{...}` 可以替换整个 optional value，但不能修改已 engaged 的
`TrtllmMlaArgs` 内部的 `const` member。

## Activation Rules

Builder 负责 activation rules，并据此 materialize optional groups：

- `kvCache is not None` iff `metadata.kv_cache_manager is not None`。
- `mla is not None` iff layer 是 MLA。Static MLA geometry 和 per-call MLA tensors 放在
  同一个 optional group，避免额外 static nesting。
  当 `mla is not None` 且 `attentionInputType != generation_only` 时，
  `TrtllmGen::is_supported(args)` 应返回 unsupported，`attention(args)` 应选择默认的
  non-TRTLLM-Gen kernel path。
- `sage is not None` iff 任一 Sage block-size 字段非零。
  `sage_attn_qk_int8` 是 Sage active 时消费的值，本身不激活 Sage。
- `sparse is not None` iff `sparse_attn_indices is not None and
  sparse_attn_indices.numel() > 0`，与当前 `_run` predicate 一致。
- `skipSoftmax is not None` iff 任一 threshold 被设置，或 block-skip stat 有意义。
  `fmha.softmaxStatsTensor` 不能激活 skip-softmax。
- `specDec is not None` iff `metadata.is_spec_decoding_enabled` 为 true。
- `helix is not None` iff `metadata.helix_position_offsets is not None`。
- `flashMla is not None` iff FlashMLA metadata 存在。

## thop.attention Interface

Python-facing `thop.attention` API 只接受一个参数：

```python
thop.attention(args: thop.TrtllmAttentionArgs) -> None
```

C++ entry point 与之保持一致：

```cpp
void attention(TrtllmAttentionArgs const& args);
```

不要在这个 entry point 后面保留第二套 hidden long-signature implementation。
`attention(args)` 应直接消费 `args`，derived values 在使用它们的代码块附近计算。

Nanobind 需要暴露：

- `TrtllmAttentionArgs`
- `TrtllmRopeArgs`
- `TrtllmQuantArgs`
- `TrtllmFmhaArgs`
- optional feature structs：`TrtllmKvCacheArgs`、`TrtllmMlaArgs`、
  `TrtllmSageArgs`、`TrtllmSparseArgs`、`TrtllmSkipSoftmaxArgs`、
  `TrtllmSpecDecArgs`、`TrtllmHelixArgs`、`TrtllmFlashMlaArgs`

Binding 规则：

- 提供包含所有 required fields 的 constructor，Python 侧优先支持 keyword arguments
- static `const` fields 用 `def_ro`
- mutable per-call fields 用 `def_rw`
- 包含 `const` fields 的 always-present substruct 整体按 read-only object 暴露；需要时只暴露其 mutable members
- optional feature groups 用 `def_rw`，便于测试替换整个 optional group

`rope` 的具体 binding 规则：top-level `TrtllmAttentionArgs::rope` 用 `def_ro`；
`TrtllmRopeArgs` 内部 static fields 用 `def_ro`；`mropeRotaryCosSin` 和
`mropePositionDeltas` 用 `def_rw`。这样 `args.rope.mropeRotaryCosSin = ...` 可以工作，
但 `args.rope = ...` 和 `args.rope.rotaryInvFreq = ...` 都会失败。

所有 direct callers 迁移后删除旧的 long nanobind binding。不要把它作为第二套 production API
保留。

`workspace` 是 `TrtllmAttentionArgs` 的 required field。C++ entry point 不分配、
不 resize workspace。

## Workspace

Python 在调用 `thop.attention(args)` 前负责选择并调整 workspace 大小。

C++ 侧暴露一个直接接收同一个 structured args object 的 pure sizing query：

```python
required_size = thop.get_attention_workspace_size(args)
```

构造 `args` 时 `args.workspace` 必须 defined，但在 sizing query 前不要求大小已经足够。
这个 query 不读取也不 resize `args.workspace`；它只是复用 normalized args，选择和
`attention(args)` 相同的 cached `AttentionOp`。

实现草图：

```cpp
int64_t getAttentionWorkspaceSize(TrtllmAttentionArgs const& args)
{
    auto const& op = getAttentionOp(args);

    int64_t const numTokens = args.q.size(0);
    bool const isGenOnly
        = args.attentionInputType == AttentionInputType::GenerationOnly;
    // ContextOnly 时 numCtxTokens == numTokens，因此 non-generation 分支返回 0。
    int64_t const numGenTokens
        = isGenOnly ? numTokens : numTokens - args.numCtxTokens;
    int64_t const maxBlocksPerSequence =
        args.kvCache.has_value() && args.kvCache->blockOffsets.has_value()
        ? args.kvCache->blockOffsets->size(-1)
        : 0;

    size_t const contextWorkspaceSize = op.getWorkspaceSizeForContext(
        op.mType, args.maxNumRequests, op.mMaxContextLength, 0, numTokens);
    size_t const generationWorkspaceSize = op.getWorkspaceSizeForGeneration(
        op.mType, args.maxNumRequests, args.maxAttentionWindowSize,
        numGenTokens, maxBlocksPerSequence);

    return std::max(contextWorkspaceSize, generationWorkspaceSize);
}
```

`getAttentionOp(args)` 是 internal C++ helper，从
`cpp/tensorrt_llm/thop/attentionOp.cpp` 中现有的 static cache 读取 op。这个 cache
仍然可以存 miss path 中一起构造的 `(AttentionOp, RunnerBase)` pair；`getAttentionOp(args)`
只返回 op，因为 workspace sizing 不需要 runner。`attention(args)` 查询同一个 cache entry，
并使用其中的 op 和 runner 执行。Workspace sizing 只调用
`AttentionOp::getWorkspaceSizeForContext` 和 `AttentionOp::getWorkspaceSizeForGeneration`，
并传入来自 `args` 的 dynamic values。第一次 query 可以 construct cached entry；
之后是 cache hit。第一版实现中，op-cache key
应该跟随现有 `AttentionOp::data()` 契约：包含存储在 `AttentionOp` 上并影响 construction
或 prepared state 的 config values，排除 tensor handles 和 `workspace`。当前仍存储在
`AttentionOp` 上的值，例如 `maskType` 或 `maxContextLength`，在后续 cleanup 将它们移出
`AttentionOp` 之前，仍属于 op-cache identity。Workspace sizing 不要新增 runner state。

Python 选择合适的 workspace tensor，用这个 handle 构造 args，查询 required size，然后在
调用 attention 前 resize 同一个 tensor。先构造 args 再 resize 是有意的：sizing query
只要求 `args.workspace` defined，不读取 buffer 内容或大小：

```python
workspace = metadata.cuda_graph_workspace if metadata.is_cuda_graph else metadata.workspace
args = thop.TrtllmAttentionArgs(..., workspace=workspace, ...)
required_size = thop.get_attention_workspace_size(args)

if metadata.is_cuda_graph:
    assert workspace.numel() >= required_size
elif workspace.numel() < required_size:
    workspace.resize_(required_size)

thop.attention(args)
```

具体 binding 细节可以在实现时决定。固定契约是：direct callers 必须调用
`get_attention_workspace_size(args)`，并在 `thop.attention(args)` 前提供 defined 且大小正确的
workspace；`attention(args)` 只能 validate。为了 defense in depth，entry point 可以复用
同一个 sizing query 做 assertion；但不能 resize 或 allocate workspace：

```cpp
// undefined tensor 上调用 .numel() 会 crash，因此先检查 defined。
TORCH_CHECK(args.workspace.defined());
int64_t const requiredWorkspaceSize = getAttentionWorkspaceSize(args);
TORCH_CHECK(args.workspace.numel() >= requiredWorkspaceSize);
```

这和当前 entry point 行为一致：当前代码每次调用也会执行同一组 `AttentionOp`
workspace-size helpers，因此 validation 不会新增 per-call sizing cost。

## Runner Interface

`RunnerBase` 不再逐个接收 attention fields，而是只接收 normalized args。Context 和
generation 拆成两个方法；phase identity 由方法名表达，因此不需要 phase struct、
`isContext` flag 或 workspace side-channel。

```cpp
class RunnerBase
{
public:
    virtual ~RunnerBase() = default;
    virtual void prepare(AttentionOp& op) const = 0;
    virtual void runContext(AttentionOp& op, TrtllmAttentionArgs const& args) const = 0;
    virtual void runGeneration(AttentionOp& op, TrtllmAttentionArgs const& args) const = 0;
};
```

Workspace sizing 有意放在 `RunnerBase` 外面。public API 是
`thop.get_attention_workspace_size(args)`；它的 C++ 实现直接调用 cached `AttentionOp`
workspace-size helpers。Runner 只负责 execution：`prepare`、context launch 和
generation launch。

每个 runner method 在使用 slicing values 的地方局部派生这些值：

示例：

- `runContext(...)` 派生 `seqOffset = 0`、`numSeqs = args.numContexts`、
  `tokenOffset = 0`、`numTokens = args.numCtxTokens` 和 context total KV len
- `runGeneration(...)` 派生 `seqOffset = args.numContexts`、
  `numSeqs = args.hostRequestTypes.size(0) - args.numContexts`、
  `tokenOffset = isGenOnly ? 0 : args.numCtxTokens`、generation token count 和
  generation total KV len；`isGenOnly` 来自
  `args.attentionInputType == AttentionInputType::GenerationOnly`
- `args.q`、`args.k`、`args.v`、`args.output`、`args.outputSf` 替代旧 signature 的 I/O 区域
- `args.sequenceLength`、`args.hostPastKeyValueLengths`、`args.contextLengths`、
  `args.hostContextLengths` 替代 batch tensors
- `args.kvCache`、`args.rope`、`args.quant`、`args.fmha` 和 optional feature groups 替代剩余长尾参数

`runContext(...)` 和 `runGeneration(...)` 内部不要在函数开头把所有 fields copy 成一套
flat local argument list。在相关代码块附近使用 local references：

```cpp
auto const& q = args.q;
auto const& rope = args.rope;
auto const& quant = args.quant;
auto const& workspace = args.workspace;

if (args.mla.has_value())
{
    auto const& mla = *args.mla;
    ...
}
```

`attention(args)` 中 context/generation 两次调用变成：

```cpp
if (args.attentionInputType != AttentionInputType::GenerationOnly)
{
    runner->runContext(*op, args);
}

if (args.attentionInputType != AttentionInputType::ContextOnly)
{
    runner->runGeneration(*op, args);
}
```

## 后续实现备注

- `TrtllmAttentionArgs` 不保存 `AttentionForwardArgs`。Builder 将每个 field 写入上面
  定义的唯一目标位置。
- TRTLLM-Gen 改为 `is_supported(op_args)` 和 `attention(op_args)`。通过 builder 解析
  KV-cache fields，从 public path 删除 `kv_cache_manager`。
- 独立的 `AttentionWorkspace` component 可以作为后续优化，不是这次 args refactor 的前置条件。
- 将 kernel selection 抽成独立 `selectFmhaKernel(args)` helper 也是后续工作。第一版实现可以
  继续把当前 selection flow inline 放在 `getAttentionOp(args)` 和 runner-construction site 中。
- Direct AutoDeploy callers 在 structured entry point 稳定后迁移：
  `_torch/auto_deploy/custom_ops/attention/trtllm_attention.py` 和
  `_torch/auto_deploy/custom_ops/mla/trtllm_mla.py`。
- 所有 direct callers 迁移后删除旧的 long-signature `thop.attention(...)` binding。
  不要把它保留成第二套 production API。
