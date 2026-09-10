# Loss-Free MoE Routing Bias Ablation

## Hypothesis

With Model Architecture, initialization, Training Batches, and Optimization held
fixed, a loss-free Routing Bias Update Rate of `0.001` should reduce Expert Load
imbalance relative to a control rate of `0` without directly adding a balancing
term to the Language Modeling Loss.

## Controlled setup

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- Seed: 1337
- Steps: 100
- Batch Shape: `(8,256)`
- Precision: BF16 Autocast
- MoE Layers: `[1,3,5]`
- Routed Experts: 8
- Top-k: 2
- Shared Experts: 1
- Expert Hidden Dimension: 352
- Control: `routing_bias_update_rate = 0`
- Treatment: `routing_bias_update_rate = 0.001`
- Both Models start with identical Parameters and receive identical Batches

## Focused results

Metrics are averaged across the three MoE Layers. Early and Late Mean CV use
the first and final ten Training Steps respectively.

| Routing Bias Rate | Early Mean CV | Late Mean CV | Cumulative CV | Zero-Selection Rate | Final Val Loss |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.5593 | 0.6492 | 0.5049 | 0.000% | 2.2232 |
| 0.001 | 0.5550 | 0.5582 | 0.4536 | 0.000% | 2.2244 |

Relative to the control, Bias updates produced:

- 14.0% lower Late Mean CV
- 10.2% lower Cumulative CV
- no Expert with zero assignments for an entire Batch in either condition
- a Validation Loss difference of only 0.0012 after 100 Steps

## Per-Layer Late Mean CV

| MoE Layer | Bias Off | Bias 0.001 |
|---|---:|---:|
| 1 | 0.541 | 0.418 |
| 3 | 0.521 | 0.434 |
| 5 | 0.886 | 0.823 |

All three Layers had lower Late Mean CV with the loss-free Bias enabled. Layer 5
remained substantially less balanced than the other Layers, so one global Bias
Update Rate does not guarantee equally balanced behavior at every depth.

## Interpretation

This Ablation provides evidence that the teaching implementation's loss-free
Routing Bias improves Expert Load Balance under the tested setup. It does not
establish that `0.001` is optimal, that Language Modeling quality is unchanged
over long Training, or that the result generalizes to other Model sizes and
Datasets.

Several Bias values reached approximately `+/-0.1` after 100 Steps. Because the
current sign-based update has no explicit clamp or decay, longer runs should
record Bias magnitude and test stability. A Rate sweep and multiple Random Seeds
are required before selecting a production Training configuration.

## Reproduce

```powershell
python experiments/ablate_moe_routing_bias.py --steps 100 --batch-size 8
```
