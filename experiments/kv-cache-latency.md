# KV Cache Latency

Date: 2026-08-11

## Question

Does reusing historical Key and Value Tensors reduce Autoregressive Generation
latency in the current Deerlight GPT?

## Correctness Requirement

Before timing, the implementation verified that:

- Single-Head cached Decode matches the corresponding Full Forward output.
- Multi-Head cached Decode matches the corresponding Full Forward output.
- Transformer Block cached Decode matches the corresponding Full Forward output.
- Model-level cached Logits match Full Forward Logits within `atol=1e-5`.
- Cached and uncached greedy Generation produce identical Token IDs.

## Controlled Variables

- Checkpoint Step: 3,000
- Validation Loss: 2.1022
- Parameters: 112,193
- Embedding Dimension: 64
- Transformer Layers: 2
- Attention Heads: 4
- Block Size: 64
- Device: CUDA
- Batch Size: 1
- Initial Context Length: 1 Token
- Generated Tokens: 60
- Sampling: Top-k 1
- Warm-up Runs: 1 per condition
- Measured Runs: 3 per condition

The 61-Token final Sequence remains within the 64-Token Context Window, so no
sliding-window Cache rebuild occurs during the measurement.

## Changed Variable

```text
use_kv_cache = False or True
```

## Results

| Condition | Total Time for 3 Runs | Mean Time per Run | Relative Speed |
|---|---:|---:|---:|
| Uncached | 0.4245 s | 0.1415 s | 1.000x |
| Cached | 0.4554 s | 0.1518 s | 0.932x |

The Cached implementation was approximately 7.3% slower in this small-model
benchmark.

## Algorithmic Work

For generated Context lengths 1 through 60, per Layer and Head:

```text
Uncached Attention Score elements: sum(t^2) = 73,810
Cached Attention Score elements:   sum(t)   = 1,830
Reduction: approximately 40.3x
```

Q/K/V Projections also process approximately 30.5x fewer Token Positions:

```text
Uncached: sum(t) = 1,830 Token Positions
Cached:              60 Token Positions
```

## Interpretation

KV Cache successfully removes repeated mathematical work, but the current Model
is too small for that reduction to dominate Wall-clock latency. Python Cache
bookkeeping, per-Head `torch.cat` operations, tiny Matrix Multiplications, and
GPU Kernel launch overhead cost more than the computation saved.

This is not evidence that KV Cache is ineffective for production LLMs. Larger
Models and longer Contexts have far more historical Projection and Attention
work to avoid, while production implementations use packed Head Dimensions,
optimized Cache storage, and fused Kernels.

## Limitations

- Only one tiny Character-level Model and GPU environment were measured.
- Batch Size was 1 and Context Length did not exceed 64.
- Cache Tensors are concatenated in Python separately for every Head.
- Only three measured runs were used.
- Peak Memory and Tokens per Second under concurrent requests were not measured.
- Learned absolute Position Embeddings require a Cache rebuild whenever the
  sliding Context Window becomes full.
