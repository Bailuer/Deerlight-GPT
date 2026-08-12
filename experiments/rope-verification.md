# RoPE Architecture Verification

Date: 2026-08-13

## Question

Can Deerlight GPT replace Learned Absolute Position Embeddings with fixed
Rotary Position Embeddings while preserving Attention, Causality, and cached
Decode correctness?

## Architecture Change

The previous input Representation was:

```text
Token Embedding + Learned Absolute Position Embedding
```

The RoPE Model now uses:

```text
Token Embedding
→ Q/K Projection in every Attention Layer
→ rotate adjacent Q/K Feature Pairs by Position-dependent Angles
```

Values are not rotated. Cached Keys are stored after RoPE has been applied, so
historical Keys do not need to be rotated again during Decode.

## Configuration

- Embedding Dimension: 64
- Query Heads: 4
- KV Heads: 4
- Head Size: 16
- Feature Pairs per Head: 8
- Transformer Layers: 2
- Context Window: 64
- RoPE Base: 10,000

## Correctness Checks

- RoPE Output Shape equals Input Shape.
- Every rotated Head Vector preserves its Euclidean Norm.
- Adding the same Position Offset to all Queries and Keys preserves the entire
  pairwise Dot-Product Score Matrix within `atol=1e-5`.
- MHA, GQA, and MQA all preserve Causality.
- Cached one-Token Decode matches Full Forward for Attention, Transformer
  Blocks, and Model Logits.
- Cached and uncached greedy Generation produce identical Token IDs.
- A RoPE Checkpoint Save/Load round-trip produces exactly equal Logits.

All checks passed.

## Parameter Count

| Architecture | Parameters |
|---|---:|
| Previous MHA with Learned Position Table | 112,193 |
| MHA with RoPE | 108,097 |

The reduction is exactly 4,096 Parameters:

```text
64 Positions × 64 Embedding Features = 4,096
```

Standard RoPE Frequencies are fixed Buffers rather than trainable Parameters.

## Checkpoint Compatibility

RoPE changes the Model function rather than only repacking Weight Tensors. The
previous trained Learned-Position Checkpoint therefore remains a historical
Baseline and is not loaded into the RoPE Model.

The RoPE Architecture uses a separate Checkpoint path:

```text
checkpoints/deerlight_gpt_rope_best.pt
```

## Limitations

- Correctness was verified on a randomly initialized RoPE Model; it has not yet
  been trained for Language Modeling quality.
- RoPE can compute Positions structurally, but the current Causal Mask and
  Training Context still limit the Model to 64 Tokens.
- Long-Context quality requires suitable Training and may require RoPE Scaling.
