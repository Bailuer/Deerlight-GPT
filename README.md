# Deerlight GPT

Learning progress, experiment evidence, review questions, and the next step are tracked in [LEARNING_LOG.md](LEARNING_LOG.md).

Building a small decoder-only Transformer from first principles, then upgrading it with modern LLM techniques.

> Status: work in progress. The first goal is a transparent character-level GPT that can train and generate text end to end.

## Goals

- Understand every important Tensor shape in a decoder-only Transformer.
- Implement the complete training path: Token IDs -> Logits -> Cross-Entropy -> Backpropagation.
- Implement Autoregressive Generation.
- Upgrade one architecture component at a time and record controlled experiments.

## Baseline Architecture

The first working version will use:

- Character-level Tokenizer
- Token Embedding
- Rotary Position Embedding (RoPE)
- Configurable packed MHA / GQA / MQA
- Residual Connections
- RMSNorm
- SwiGLU Feed-Forward Network
- Vocabulary Projection
- Cross-Entropy Loss
- AdamW
- Optional per-Layer, per-Head KV Cache for Autoregressive Generation

## Roadmap

### Baseline Deerlight GPT

- [x] Character-level Tokenizer
- [x] Training and Validation Split
- [x] Shifted Training Batches
- [x] Bigram Language Model
- [x] Single-Head Self-Attention
- [x] Multi-Head Self-Attention
- [x] Transformer Block
- [x] Decoder-only GPT
- [x] Training Loop
- [x] Autoregressive Generation

### Modern LLM Upgrades

- [x] KV Cache
- [x] MQA / GQA
- [x] RoPE
- [x] RMSNorm
- [x] SwiGLU
- [x] PyTorch SDPA / Fused cuDNN Attention
- [ ] FlashAttention-2 Kernel (not included in the tested Windows PyTorch Wheel)
- [ ] Mixture of Experts
- [ ] Scaling Law Experiments

## Project Structure

```text
Deerlight-GPT/
|-- deerlight_gpt.py      # First transparent single-file implementation
|-- train_medium.py       # Reproducible CUDA Pilot and Medium training entry point
|-- data/                 # Local training text (not committed by default)
|-- notes/                # Concepts rewritten in the author's own words
|-- experiments/          # Controlled experiments and results
|-- assets/               # Images used by the README
|-- requirements.txt
|-- .gitignore
`-- README.md
```

The first version intentionally stays in one Python file. It will be refactored into Modules only after the complete data flow works.

## Learning Rules

1. Every new Module must have its input and output shapes documented.
2. Every optimization must be compared with a Baseline.
3. Training Loss alone is not enough; Validation Loss, speed, and memory should be recorded when relevant.
4. Notes should be rewritten from understanding instead of copied from conversations or tutorials.
5. Unfinished features remain unchecked in the Roadmap.

## Dataset

Place a licensed or self-authored text corpus at:

```text
data/input.txt
```

The Dataset itself is ignored by Git by default. Record its source and license in `data/README.md` before publishing experimental results.

## Run

```powershell
python deerlight_gpt.py
```

Run a 30-Step resource and stability Pilot before committing to a full
Medium experiment:

```powershell
python train_medium.py --mode pilot
```

Then train the 9.49M-Parameter Medium capability baseline:

```powershell
python train_medium.py --mode train
```

The Medium script uses BF16 Autocast, fused AdamW, and fused cuDNN Attention on
the tested CUDA environment. It clips the Gradient Norm, applies Early
Stopping, and generates from the Best Validation Checkpoint rather than the
final training state. Use `--attention-backend manual` to retain the transparent
Attention path or `--attention-backend sdpa` to inspect automatic SDPA dispatch.

The training script automatically uses CUDA when a CUDA-enabled PyTorch build
is available, and otherwise falls back to CPU.

For the tested Windows + NVIDIA setup, install the CUDA-enabled PyTorch wheel
inside the Virtual Environment before running Training:

```powershell
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
```

## Current Milestone

Milestone 16 complete: a transparent Sparse MoE Feed-Forward layer now supports
Top-k Token Routing, fine-grained SwiGLU Routed Experts, one always-active Shared
Expert, loss-free Routing Bias updates, Expert Load monitoring, and weighted
Scatter-Add combination. MoE can be enabled only on selected Transformer Blocks,
while the default Dense path remains compatible with existing Checkpoints.

The first matched-Active-Capacity Pilot found 78.1% more Total Parameters and
only 1.3% more estimated Active Parameters per Token, but the transparent MoE
implementation delivered 57.4% lower throughput and 33.6% higher Peak VRAM than
Dense. Thirty Steps were insufficient to establish a quality difference or
improved Expert balance.

A 100-Step MoE-vs-MoE Ablation then isolated loss-free Routing Bias updates.
Rate `0.001` reduced cross-Layer Late Mean Load CV by 14.0% and Cumulative CV
by 10.2% relative to Rate `0`, with only a 0.0012 Validation Loss difference.

Next: sweep multiple Routing Bias Update Rates and Random Seeds, or return to
the Modern Architecture curriculum and study Scaling Laws before committing to
a longer MoE training run.

The expanded Mini IsoFLOP Profile held approximate Dense Training Compute near
`1.2e12 FLOPs` across five Model sizes. Validation Loss formed a U-shaped curve:
the 55,793-Parameter Mini Model reached 2.0151, outperforming both a
capacity-limited 16,641-Parameter Tiny Model and increasingly undertrained Models
from 101K to 723K Parameters. This brackets a local Compute-Optimal region for
the tested budget without claiming a universal optimum.
