# Medium Capability Baseline

## Question

Can the modern Deerlight GPT Architecture produce recognizably structured
Shakespeare-like text when scaled from the tiny verification model to a still
interpretable Medium configuration? How much CUDA memory and training time does
that configuration require on the available laptop GPU?

## Reproducible Configuration

- Dataset: Tiny Shakespeare, character-level Vocabulary of 65 Tokens
- Training split: 1,003,854 Tokens
- Random seed: 1337
- Embedding dimension (`C`): 384
- Transformer Layers: 6
- Query Heads: 6
- KV Heads: 2 (GQA)
- Head dimension: 64
- Context length (`T`): 256
- SwiGLU hidden dimension: 1024
- Parameters: 9,494,465
- Batch Size: 32
- Tokens per Step: 8,192
- Optimizer: fused AdamW
- Learning Rate: 3e-4, constant
- Precision: FP32 Parameters with BF16 Autocast Activations
- Gradient Clipping: maximum norm 1.0
- Evaluation: 20 randomly sampled Batches every 250 Steps
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU

## Resource Pilot

A 30-Step Pilot was run before the full experiment.

| Metric | Result |
|---|---:|
| Initial Validation Loss | 4.3501 |
| Step 30 Validation Loss | 2.5235 |
| Measured throughput | 270,304 Tokens/s |
| Peak VRAM allocated | 1.676 GiB |
| Peak VRAM reserved | 1.828 GiB |

The Batch Size fit comfortably, all Loss values remained finite, and the raw
Gradient Norm fell from 2.3239 at Step 1 to roughly 0.4-0.5 during the Pilot.

## Full Training Result

| Step | Train Loss | Validation Loss | Interpretation |
|---:|---:|---:|---|
| 0 | 4.3503 | 4.3533 | Random initialization |
| 250 | 1.5817 | 1.7621 | Rapid shared learning |
| 500 | 1.3583 | 1.5911 | Generalization improving |
| 750 | 1.2600 | 1.5616 | Near the best region |
| 1000 | 1.1621 | **1.5610** | Best saved Checkpoint |
| 5000 | 0.1647 | 3.8398 | Severe Overfitting |

The 5000-Step run processed 40.96M sampled training Tokens. Excluding the first
five warmup Steps, measured training time was 159.018 seconds and throughput
was 257,324 Tokens/s. Peak allocated VRAM remained 1.676 GiB.

## Best-Checkpoint Sample

Sampling used Temperature 0.8 and Top-k 20 from the Step 1000 Checkpoint:

```text
COMINIUS:
O, what's a sister
That thunder we are rude the gates of man
The sword his back that affection with fall,
And to heavens, to thy heart and my life
As if the necessity to claim the deed.

GLOUCESTER:
Welcome, dispatch; she could you speak is here,
That I shall do this deadly thus demand?
```

The sample has learned character names, dialogue formatting, punctuation, and
local Shakespeare-like syntax. Its global semantics remain unreliable, so this
is a language-structure baseline rather than a generally capable language model.

## Interpretation

The experiment demonstrates three different facts:

1. The Architecture and optimizer can learn the Dataset stably.
2. Increased capacity produces far more coherent surface structure than the
   earlier 31K-Parameter model.
3. More optimization Steps are actively harmful after the best generalization
   region on this small corpus. Falling Training Loss does not prove that the
   model is improving.

This is not a clean Scaling Law experiment because Architecture, width, depth,
context length, precision, and training budget changed together. It is a
capability and resource baseline. Future training should use Early Stopping;
Learning Rate Decay, regularization, and more diverse data are separate
controlled variables to test rather than assumptions.
