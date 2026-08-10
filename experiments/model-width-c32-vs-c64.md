# Model Width: C=32 vs C=64

Date: 2026-08-10

## Question

Does increasing only the Embedding Dimension improve the Validation Loss and
generation quality of the baseline Deerlight GPT?

## Controlled Variables

- Dataset: Tiny Shakespeare, 1,115,394 Characters
- Vocabulary Size: 65
- Random Seed: 1337
- Batch Size: 32
- Block Size: 64
- Number of Heads: 4
- Number of Layers: 2
- Feed-Forward Expansion: 4
- Learning Rate: 3e-4
- Training Steps: 3,000
- Evaluation Interval: 300 Steps
- Evaluation Batches: 20
- Device: NVIDIA GeForce RTX 5080 Laptop GPU
- PyTorch: 2.12.1+cu130

## Changed Variable

```text
EMBEDDING_DIM: 32 -> 64
```

With four Attention Heads, this also changes each Head Size from 8 to 16.

## Results

| Metric | C=32 Baseline | C=64 Experiment |
|---|---:|---:|
| Parameters | 31,553 | 112,193 |
| Final Training Loss | 2.3305 | 2.0584 |
| Final Validation Loss | 2.3632 | 2.1022 |
| Validation Perplexity | 10.62 | 8.18 |
| Train-Validation Gap | 0.0327 | 0.0438 |

Increasing Model Width reduced Validation Loss by 0.2610 and reduced
Validation Perplexity by approximately 23%.

## Qualitative Result

The C=32 model learned English-like Character patterns and punctuation but
rarely produced stable words or sentence fragments.

The C=64 model produced more recognizable words, phrases, speaker-like labels,
and Shakespeare-style sentence structure, although it remained semantically
incoherent.

## Interpretation

The wider model used the same Dataset and Optimization Budget more effectively.
Its Training and Validation Losses were both still decreasing at Step 3,000,
and the small Generalization Gap provided no evidence of serious Overfitting.

The result supports the hypothesis that the 31K-Parameter Baseline was limited
by Model Capacity. The C=64 configuration is now the default Deerlight GPT
Baseline for the next experiment.

## Limitations

- Only one Random Seed was tested.
- Training speed and peak VRAM were not recorded.
- Generation quality was assessed qualitatively rather than with a formal Metric.
- Character-level Tokenization limits word-level consistency.
