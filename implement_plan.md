<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
-->

# TrtllmAttentionArgs Args Design

Scope: this document defines the args classification, struct shape, and the new
`thop.attention` / `Runner` interface shape. The staged migration,
direct-caller updates, validation matrix, and benchmarking plan should be
written after this design is accepted.

## Goal

`thop.attention` and TRTLLM-Gen should consume the same attention argument
contract. Before changing call sites, first define a clear `TrtllmAttentionArgs`
classification that is simple, flexible, and cheap to build.

Design rule:

- Keep core attention fields flat when nesting does not add meaning.
- Use a substruct only when it has a real subsystem boundary or feature
  activation meaning.
- Use `const` members for layer-static fields so the type records that they are
  construction-time state.
- Keep per-call fields mutable for builder ergonomics and targeted tests.

## Classification

The final top-level shape should be mostly flat:

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

Avoid `io`, `batch`, and `layer` substructs for now. Those groups mostly repeat
the existing argument categories and make call sites more verbose without
creating a stronger invariant.

## Struct Shape

### RoPE

RoPE has enough local state to justify a substruct. Static RoPE fields are
`const`; mRoPE tensors are per-call.

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

These are resolved per-call values. The builder picks the forward override when
present, otherwise the layer default.

```cpp
struct TrtllmQuantArgs
{
    std::optional<at::Tensor> kvScaleOrigQuant;
    std::optional<at::Tensor> kvScaleQuantOrig;
    std::optional<at::Tensor> outScale;
};
```

### FMHA

`softmaxStatsTensor` belongs here. It is an FMHA softmax-stats output buffer
used by Helix and MLA merge paths. It is not a skip-softmax feature flag.

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

Feature groups are optional. `std::nullopt` means the feature is inactive.

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

## Const And Construction

The `const` fields intentionally delete whole-object assignment for structs that
contain static state. That is acceptable if the builder uses one-shot
construction:

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

Nanobind bindings should expose static fields with `def_ro` and per-call fields
with `def_rw`. For substructs containing `const` members, prefer constructors
and read-only exposure of static fields instead of relying on whole-substruct
replacement.

Always-present substructs that contain `const` members, such as `rope`, must be
initialized in the top-level construction expression; they cannot be assigned
post-hoc. Optional feature groups are different: `args.mla = TrtllmMlaArgs{...}`
can replace the optional value, but mutating a `const` member inside the engaged
`TrtllmMlaArgs` is not allowed.

## Activation Rules

The builder owns activation rules and materializes optional groups accordingly:

- `kvCache is not None` iff `metadata.kv_cache_manager is not None`.
- `mla is not None` iff the layer is MLA. Static MLA geometry lives in the same
  optional group as the per-call MLA tensors to avoid an extra static nesting
  layer.
  When `mla is not None` and `attentionInputType != generation_only`,
  `TrtllmGen::is_supported(args)` should return unsupported and
  `attention(args)` should select the default non-TRTLLM-Gen kernel path.
- `sage is not None` iff any Sage block-size field is nonzero.
  `sage_attn_qk_int8` is consumed when Sage is active; it does not activate Sage
  by itself.
- `sparse is not None` iff `sparse_attn_indices is not None and
  sparse_attn_indices.numel() > 0`, matching the current `_run` predicate.
- `skipSoftmax is not None` iff either threshold is set or block-skip stat is
  meaningful. `fmha.softmaxStatsTensor` must not activate skip-softmax.
- `specDec is not None` iff `metadata.is_spec_decoding_enabled` is true.
- `helix is not None` iff `metadata.helix_position_offsets is not None`.
- `flashMla is not None` iff FlashMLA metadata is present.

## thop.attention Interface

The Python-facing `thop.attention` API should accept exactly one argument:

```python
thop.attention(args: thop.TrtllmAttentionArgs) -> None
```

The C++ entry point mirrors it:

```cpp
void attention(TrtllmAttentionArgs const& args);
```

Do not keep a second long-signature implementation hidden behind this entry
point. `attention(args)` should consume `args` directly and compute derived
values near the code that uses them.

Nanobind should expose:

- `TrtllmAttentionArgs`
- `TrtllmRopeArgs`
- `TrtllmQuantArgs`
- `TrtllmFmhaArgs`
- optional feature structs: `TrtllmKvCacheArgs`, `TrtllmMlaArgs`,
  `TrtllmSageArgs`, `TrtllmSparseArgs`, `TrtllmSkipSoftmaxArgs`,
  `TrtllmSpecDecArgs`, `TrtllmHelixArgs`, and `TrtllmFlashMlaArgs`

Binding rules:

- provide constructors that take all required fields, preferably with keyword
  arguments on the Python side
- expose static `const` fields with `def_ro`
- expose mutable per-call fields with `def_rw`
- expose always-present substructs containing `const` fields as read-only
  whole objects, with mutable access only to their mutable members when needed
- expose optional feature groups with `def_rw`, so tests can replace the whole
  optional group

Concrete binding rule for `rope`: bind `TrtllmAttentionArgs::rope` with
`def_ro`, bind static fields inside `TrtllmRopeArgs` with `def_ro`, and bind
`mropeRotaryCosSin` / `mropePositionDeltas` with `def_rw`. This allows
`args.rope.mropeRotaryCosSin = ...` but rejects `args.rope = ...` and
`args.rope.rotaryInvFreq = ...`.

The old long nanobind binding should be removed after all direct callers are
migrated. It should not remain as a parallel production API.

`workspace` is a required field on `TrtllmAttentionArgs`. The C++ entry point
does not allocate or resize workspace.

## Workspace

Python owns workspace selection and sizing before it calls
`thop.attention(args)`.

The C++ side should expose a pure sizing query that takes the same structured
args object:

```python
required_size = thop.get_attention_workspace_size(args)
```

The `args.workspace` tensor must be defined when `args` is constructed, but it
does not need to be large enough before the sizing query. The query does not
read or resize `args.workspace`; it only reuses the normalized args to select
the same cached `AttentionOp` as `attention(args)`.

Implementation sketch:

```cpp
int64_t getAttentionWorkspaceSize(TrtllmAttentionArgs const& args)
{
    auto const& op = getAttentionOp(args);

    int64_t const numTokens = args.q.size(0);
    bool const isGenOnly
        = args.attentionInputType == AttentionInputType::GenerationOnly;
    // For ContextOnly, numCtxTokens == numTokens, so the non-generation
    // branch returns 0.
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

`getAttentionOp(args)` is the internal C++ helper that reads the op from the
existing static cache in `cpp/tensorrt_llm/thop/attentionOp.cpp`. The cache can
still store the `(AttentionOp, RunnerBase)` pair built together on the miss
path; `getAttentionOp(args)` returns only the op because workspace sizing does
not need the runner. `attention(args)` looks up the same cache entry and uses
both the op and runner for execution. Workspace sizing only calls
`AttentionOp::getWorkspaceSizeForContext` and
`AttentionOp::getWorkspaceSizeForGeneration` with dynamic values from `args`.
The first query can construct the cached entry; later calls are cache hits.
For the first implementation, the op-cache key should follow the existing
`AttentionOp::data()` contract: include config values that are stored on
`AttentionOp` and affect construction or prepared state, and exclude tensor
handles and `workspace`. Values that are currently stored on `AttentionOp`, such
as `maskType` or `maxContextLength`, remain part of the op-cache identity until
they are moved out of `AttentionOp` in a later cleanup. Do not add runner state
to workspace sizing.

Python chooses the right workspace tensor, builds args with that handle, queries
the required size, then resizes the same tensor before calling attention.
Building args before resizing is intentional because the sizing query only
requires `args.workspace` to be defined; it does not read the buffer contents or
size:

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

The exact binding details can be decided during implementation. The contract is
fixed: direct callers must call `get_attention_workspace_size(args)` and provide
a defined, correctly sized workspace before `thop.attention(args)`;
`attention(args)` may only validate it. For defense in depth, the entry point
can reuse the same sizing query for the assertion; it must not resize or
allocate workspace:

```cpp
// Calling .numel() on an undefined tensor crashes, so check defined first.
TORCH_CHECK(args.workspace.defined());
int64_t const requiredWorkspaceSize = getAttentionWorkspaceSize(args);
TORCH_CHECK(args.workspace.numel() >= requiredWorkspaceSize);
```

This matches the current entry-point behavior, which already calls
the same `AttentionOp` workspace-size helpers on every invocation, so the
validation does not add a new per-call sizing cost.

## Runner Interface

`RunnerBase` should stop accepting attention fields individually. It should take
the normalized args only. Context and generation are separate methods; the
method name carries the phase identity, so no phase struct, `isContext` flag, or
workspace side-channel is needed.

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

Workspace sizing is intentionally outside `RunnerBase`. The public API is
`thop.get_attention_workspace_size(args)`, and its C++ implementation calls the
cached `AttentionOp` workspace-size helpers directly. Runners own execution
only: `prepare`, context launch, and generation launch.

Each runner method derives its own slicing values locally where they are used:

Examples:

- `runContext(...)` derives `seqOffset = 0`, `numSeqs = args.numContexts`,
  `tokenOffset = 0`, `numTokens = args.numCtxTokens`, and context total KV len
- `runGeneration(...)` derives `seqOffset = args.numContexts`,
  `numSeqs = args.hostRequestTypes.size(0) - args.numContexts`,
  `tokenOffset = isGenOnly ? 0 : args.numCtxTokens`, generation token count,
  and generation total KV len; `isGenOnly` is derived from
  `args.attentionInputType == AttentionInputType::GenerationOnly`
- `args.q`, `args.k`, `args.v`, `args.output`, and `args.outputSf` replace the
  I/O section of the old signature
- `args.sequenceLength`, `args.hostPastKeyValueLengths`, `args.contextLengths`,
  and `args.hostContextLengths` replace the batch tensors
- `args.kvCache`, `args.rope`, `args.quant`, `args.fmha`, and optional feature
  groups replace the remaining long tail

Inside `runContext(...)` and `runGeneration(...)`, avoid copying fields into a
flat local argument list at the top of the function. Use local references near
the relevant block:

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

The context and generation calls in `attention(args)` become:

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

## Notes For The Later Implementation

- `AttentionForwardArgs` is not stored in `TrtllmAttentionArgs`. The builder
  writes each field once into the concrete destination above.
- TRTLLM-Gen should take `is_supported(op_args)` and `attention(op_args)`.
  Remove `kv_cache_manager` from that public path by resolving KV-cache fields
  in the builder.
- A standalone `AttentionWorkspace` component is a possible follow-up, not a
  prerequisite for this args refactor.
- Extracting kernel selection into a dedicated `selectFmhaKernel(args)` helper
  is also a follow-up. The first implementation can keep the current selection
  flow inline inside `getAttentionOp(args)` and the runner-construction site.
- Direct AutoDeploy callers migrate after the structured entry point is stable:
  `_torch/auto_deploy/custom_ops/attention/trtllm_attention.py` and
  `_torch/auto_deploy/custom_ops/mla/trtllm_mla.py`.
- Remove the old long-signature `thop.attention(...)` binding after all direct
  callers have migrated. It should not remain as a second production API.
