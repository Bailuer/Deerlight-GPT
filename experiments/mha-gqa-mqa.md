# MHA vs GQA vs MQA

Date: 2026-08-11

## Question

How does reducing the number of Key/Value Heads change Parameter count,
persistent KV Cache size, and cached Generation latency in Deerlight GPT?

## Controlled Variables

- Query Heads: 4
- Embedding Dimension: 64
- Head Size: 16
- Transformer Layers: 2
- Block Size: 64
- Vocabulary Size: 65
- Data Type: float32
- Device: CUDA
- Batch Size: 1
- Initial Context Length: 1 Token
- Generated Tokens: 60
- Sampling: Top-k 1
- Warm-up Runs: 2
- Measured Runs: 10
- Seed: 1337, reset before creating each Model

All Models were randomly initialized. This experiment measures Architecture
costs, not trained Language Modeling quality.

## Changed Variable

| Architecture | Query Heads | KV Heads | Queries per KV Head |
|---|---:|---:|---:|
| MHA | 4 | 4 | 1 |
| GQA | 4 | 2 | 2 |
| MQA | 4 | 1 | 4 |

## Correctness Checks

For MHA, GQA, and MQA independently:

- Full Forward Output Shape is `(B,T,C)`.
- Future Tokens cannot affect earlier Outputs.
- Cached one-Token Decode matches the corresponding Full Forward Output.
- Cache Shape is `(B,num_kv_heads,T,head_size)`.

The legacy trained MHA Checkpoint was also migrated from independent Head
Weights to packed Projection Weights. Its migrated Logits had a maximum
absolute difference of `0.0` from the pre-refactor Reference Logits.

## Results

| Architecture | KV Heads | Parameters | KV Cache Bytes | Mean Latency |
|---|---:|---:|---:|---:|
| MHA | 4 | 112,193 | 65,536 | 76.003 ms |
| GQA | 2 | 104,001 | 32,768 | 80.506 ms |
| MQA | 1 | 99,905 | 16,384 | 83.146 ms |

Relative to MHA:

- GQA uses 50% of the persistent KV Cache and 7.3% fewer Parameters.
- MQA uses 25% of the persistent KV Cache and 11.0% fewer Parameters.
- GQA was 5.9% slower and MQA was 9.4% slower in this tiny-model benchmark.

## Interpretation

The KV Cache reduction exactly follows the number of KV Heads. Query Heads and
the final Attention Output width remain unchanged, while only the Key and Value
Projection widths shrink.

Wall-clock latency did not improve at this scale. The transparent PyTorch
implementation uses `repeat_interleave` to physically map shared K/V Heads to
all Query Heads before Attention. For this tiny Model, that Tensor operation and
GPU Kernel launch overhead exceed the saved K/V Projection work.

Production GQA Kernels operate on grouped Heads without materializing a repeated
K/V Tensor. The current result therefore validates persistent Cache capacity,
but it is not a production Throughput estimate.

## Limitations

- The GQA and MQA Models were not trained, so quality was not compared.
- Persistent Cache bytes were measured; peak temporary GPU Memory was not.
- The implementation physically expands K/V for Attention computation.
- Only one small Model, Batch Size, Context Length, and GPU were tested.
- Latency results may be dominated by Python and Tiny Kernel overhead.

## Reproduction

```powershell
python experiments/benchmark_gqa.py
```
