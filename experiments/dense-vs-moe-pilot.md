# Dense vs Interleaved Sparse MoE Pilot

## Hypothesis

With approximately matched per-Token Active Parameters, an Interleaved Sparse
MoE model should expose substantially more Total Parameters than the Dense
baseline. The transparent teaching implementation is expected to trade
throughput and memory efficiency for inspectable Routing behavior.

## Controlled setup

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- Seed: 1337
- Steps: 30
- Batch Shape: `(8,256)`
- Precision: BF16 Autocast
- Dense Model: six Dense SwiGLU Transformer Blocks
- MoE Model: Layers `[1,3,5]` use Sparse MoE
- Routed Experts: 8
- Top-k: 2
- Shared Experts: 1
- Expert Hidden Dimension: 352
- Routing Bias Update Rate: 0.001
- All Shape-compatible Dense and MoE Parameters share identical initial values
- Both Models receive the same pre-generated Training and Validation Batches

The Dense hidden width is 1024. Each MoE Token activates two Routed Experts and
one Shared Expert, giving an Active Hidden Capacity of `3 * 352 = 1056`.

## Results

| Model | Total Params | Active Params/Token | Tokens/s | Peak VRAM GiB | Initial Val Loss | Final Train Loss | Final Val Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 9,494,465 | 9,494,465 | 71,291 | 0.452 | 4.3519 | 2.5814 | 2.5991 |
| Interleaved MoE | 16,913,345 | 9,614,273 | 30,358 | 0.604 | 4.3606 | 2.5791 | 2.5984 |

Relative to Dense, Interleaved MoE had:

- 78.1% more Total Parameters
- 1.3% more estimated per-Token Active Parameters
- 57.4% lower measured throughput
- 33.6% higher Peak Allocated VRAM
- no meaningful Validation Loss difference after only 30 Steps

## Expert utilization

| MoE Layer | Cumulative Max/Mean | Cumulative CV | First-Batch CV | Last-Batch CV |
|---|---:|---:|---:|---:|
| 1 | 1.596 | 0.405 | 0.334 | 0.433 |
| 3 | 1.513 | 0.341 | 0.386 | 0.400 |
| 5 | 1.671 | 0.395 | 0.363 | 0.472 |

All assignment-count invariants passed: every MoE Layer recorded exactly
`Steps * B * T * Top-k` Routed assignments. However, Last-Batch CV did not
improve over First-Batch CV in this short run.

## Interpretation

The experiment demonstrates the defining MoE tradeoff: substantially more
stored Parameters at nearly matched per-Token Active Capacity. It does not show
a quality improvement because 30 Steps are only sufficient for a resource and
correctness Pilot.

The transparent Python Expert loop, dynamic Gather operations, small per-Expert
matrix multiplications, and Scatter-Add make this implementation much slower
than the fused Dense FFN. This is an implementation result, not evidence that
optimized large-scale MoE systems are inherently slower.

The Routing Bias result is inconclusive. Router Parameters, Token Batches, and
Bias all changed during Training, and the final Batch was not more balanced than
the first. The next controlled experiment must hold the MoE Architecture fixed
and compare Bias Update Rate `0` against `0.001` across a longer run.

## Reproduce

```powershell
python experiments/benchmark_moe.py --steps 30 --batch-size 8
```
