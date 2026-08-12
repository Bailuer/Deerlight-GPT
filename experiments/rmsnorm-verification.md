# RMSNorm Architecture Verification

Date: 2026-08-13

## Question

Can a custom RMSNorm replace every LayerNorm in Deerlight GPT while preserving
stable Tensor Scale, Gradient Flow, Attention causality, and cached Generation?

## Architecture Change

The previous Pre-Norm branches used:

```text
x + Attention(LayerNorm(x))
x + FeedForward(LayerNorm(x))
```

The new branches use:

```text
x + Attention(RMSNorm(x))
x + FeedForward(RMSNorm(x))
```

The Final LayerNorm before Vocabulary Projection is also replaced by RMSNorm.

For each Token Vector, RMSNorm computes:

```text
mean_square = mean(x²)
inverse_rms = rsqrt(mean_square + epsilon)
output = weight × x × inverse_rms
```

It does not subtract the Feature Mean and has no trainable Bias.

## Configuration

- Embedding Dimension: 64
- Transformer Layers: 2
- RMSNorm Modules: 5
- Epsilon: 1e-5
- Trainable Parameters per RMSNorm: 64

## Correctness Checks

- Output Shape equals Input Shape.
- With Scale initialized to one, every normalized Token has RMS approximately
  equal to one.
- Multiplying an Input Token Vector by positive 10 produces the same normalized
  Output within `atol=1e-4`.
- The RMSNorm Weight and Input Tensor receive finite Gradients.
- The Model contains five RMSNorm Modules and no `nn.LayerNorm` Modules.
- RoPE, MHA/GQA/MQA, Causality, KV Cache, and cached-vs-uncached Generation
  checks continue to pass.
- Checkpoint Save/Load produces exactly equal Logits.

All checks passed.

## Parameter Count

| Architecture | Total Model Parameters | Norm Parameters |
|---|---:|---:|
| RoPE + LayerNorm | 108,097 | 640 |
| RoPE + RMSNorm | 107,777 | 320 |

The five RMSNorm Modules remove 320 Parameters in total because each one keeps
only a 64-element trainable Scale instead of both Scale and Bias.

## Checkpoint Compatibility

RMSNorm changes both the computation and Norm Parameter schema. The new Model
uses a separate Checkpoint path:

```text
checkpoints/deerlight_gpt_rope_rms_best.pt
```

Previous Architecture Checkpoints remain historical Baselines.

## Limitations

- The RMSNorm Model has not yet been trained for Language Modeling quality.
- Scale invariance was checked for positive scaling; negative scaling flips the
  normalized Vector direction as expected.
- Tiny-model Parameter savings do not measure production Training Throughput.
