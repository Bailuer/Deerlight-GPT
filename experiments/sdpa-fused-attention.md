# SDPA and Fused Attention

## Question

Can PyTorch Scaled Dot-Product Attention replace the transparent manual
Attention implementation without changing its Forward, Backward, GQA, or KV
Cache behavior? Which fused backend is actually available on the tested Windows
CUDA environment, and what speed and memory benefit does it provide?

## Mathematical Equivalence

The manual path explicitly computes:

```text
scores  = (Q @ K.transpose(-2, -1)) / sqrt(head_size)
scores  = causal_mask(scores)
weights = softmax(scores)
output  = weights @ V
```

The SDPA path delegates the same operation to
`torch.nn.functional.scaled_dot_product_attention`. Tests copy identical
Weights into the two implementations and compare:

- MHA, GQA, and MQA Forward Outputs
- Input Gradients and every Attention Parameter Gradient
- Full Causal Prefill
- Multiple new Tokens with a Position-offset KV Cache Mask
- Single-Token Cached Decode

All CPU FP32 equivalence checks passed within explicit floating-point
tolerances. The model Parameter count and Checkpoint Parameter Shapes do not
change when switching Backends.

## Environment Discovery

- PyTorch: 2.12.1+cu130
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- Input precision: BF16 under CUDA Autocast
- Shape: `(B,T,C) = (32,256,384)`
- Query Heads / KV Heads: 6 / 2

Although `torch.backends.cuda.flash_sdp_enabled()` returned `True`, forcing the
Flash backend failed with:

```text
Torch was not compiled with flash attention.
```

The flag only indicates that dispatch is enabled; it does not prove the PyTorch
Binary contains an executable FlashAttention-2 Kernel. For this Shape,
automatic SDPA dispatch was inspected with PyTorch Profiler and selected
`aten::_scaled_dot_product_attention_math`. The installed Build does contain a
working `CUDNN_ATTENTION` Backend, which passed forced BF16 GQA Forward and
Backward tests.

## Attention-Module Benchmark

The controlled Benchmark runs one GQA Attention Module with BF16 Forward and
Backward. It uses 10 warmup Steps and 50 measured Steps.

| Backend | Mean Forward+Backward | Peak Allocated VRAM |
|---|---:|---:|
| Manual | 2.831 ms | 0.294 GiB |
| SDPA Math | 3.336 ms | 0.311 GiB |
| SDPA cuDNN | 1.939 ms | 0.169 GiB |

Relative to Manual, fused cuDNN Attention was approximately 1.46x faster and
used 42.5% less Peak Allocated VRAM for the isolated Module. SDPA Math was
slower than Manual, demonstrating that the API name alone is not an
optimization.

## End-to-End Medium Pilot

The 9.49M-Parameter Medium model was trained for 30 Steps with identical Seed,
Batch, Context, BF16 precision, and optimizer settings.

| Backend | Tokens/s | Peak Allocated VRAM | Step 30 Validation Loss |
|---|---:|---:|---:|
| Manual | 257,158 | 1.676 GiB | 2.5236 |
| Automatic SDPA (Math) | 244,809 | 1.637 GiB | 2.5235 |
| Forced SDPA cuDNN | 343,833 | 1.245 GiB | 2.5236 |

Forced cuDNN increased end-to-end throughput by 33.7% and reduced Peak
Allocated VRAM by 25.7% relative to Manual. The nearly identical Loss and
Gradient trajectories support mathematical equivalence at training precision.

## Implementation Boundary

`sdpa_cudnn` is forced only when all tested kernel conditions hold: CUDA,
FP16/BF16 Q/K/V, square Training or Prefill, and no past KV Cache. FP32
Generation and Cached Decode use the general SDPA route so unsupported Shapes
cannot fail with `No available kernel`.

The transparent `manual` Backend remains available for education and reference.
`sdpa` remains available for testing PyTorch's automatic dispatch. The Medium
training default is `sdpa_cudnn` for this tested environment.

## Limitations

- Results apply to this GPU, PyTorch Build, precision, and Shape.
- The experiment verifies fused cuDNN Attention, not FlashAttention-2.
- Short Pilot timing still contains system noise; the isolated Module Benchmark
  uses more repetitions to stabilize comparison.
- Longer contexts should produce a larger memory advantage because manual
  Attention materializes matrices that scale with `T^2`.
