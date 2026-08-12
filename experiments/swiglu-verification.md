# SwiGLU Architecture Verification

Date: 2026-08-13

## Question

Can a bias-free SwiGLU Feed-Forward Network replace the GELU MLP while keeping
approximately the same Parameter budget and preserving end-to-end correctness?

## Architecture Change

The previous Feed-Forward Network was:

```text
Linear(C,4C) → GELU → Linear(4C,C)
```

The SwiGLU Network uses two independent Input Projections:

```text
gate  = SiLU(GateProjection(x))
value = UpProjection(x)
hidden = gate × value
output = DownProjection(hidden)
```

All three Linear Projections are bias-free. Multiplication between the Gate and
Value paths is element-wise.

## Hidden Dimension

A SwiGLU Layer has approximately `3CD` Weight Parameters. To remain close to
the original GELU MLP budget of approximately `8C²`, the target width is:

```text
D ≈ 8C/3
```

The implementation rounds this width upward to a multiple of 16:

```text
C=64  → target=170.67 → D=176
C=384 → target=1024   → D=1024
```

The resolved `feed_forward_dim` is stored explicitly in every Checkpoint rather
than inferred from an expansion ratio.

## Correctness Checks

- SwiGLU Output Shape equals Input Shape `(B,T,C)`.
- Gate and Up Projections use independent Weight Tensors.
- Parameter count equals `3 × C × D` for the bias-free implementation.
- Input, Gate, Up, and Down Tensors all receive finite Gradients.
- Transformer Block, RoPE, RMSNorm, MHA/GQA/MQA, Causality, and cached Decode
  checks continue to pass.
- Cached and uncached greedy Generation produce identical Token IDs.
- Checkpoint Save/Load produces exactly equal Logits and restores `D=176`.

All checks passed.

## Parameter Count

| Feed-Forward Architecture | Parameters per Layer |
|---|---:|
| GELU with `C→4C→C` and Bias | 33,088 |
| SwiGLU with `C→176→C`, three bias-free Projections | 33,792 |

The aligned SwiGLU Layer uses 704 more Parameters, approximately 2.1%, per
Layer. This is a small alignment cost compared with retaining `D=4C`, which
would increase the leading Weight budget from `8C²` to `12C²`.

The complete two-Layer Tiny Model now has 109,185 Parameters.

## Checkpoint Compatibility

SwiGLU changes the Feed-Forward Parameter schema. The new Architecture uses:

```text
checkpoints/deerlight_gpt_rope_rms_swiglu_best.pt
```

Previous Architecture Checkpoints remain historical Baselines.

## Limitations

- The SwiGLU Model has not yet been trained for Language Modeling quality.
- Hardware alignment does not guarantee a measurable speedup in this tiny
  Python implementation.
- Training quality must be compared using controlled, separately trained Models.
