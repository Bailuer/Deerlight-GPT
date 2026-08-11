# Deerlight GPT

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
- Learned Position Embedding
- Causal Multi-Head Self-Attention
- Residual Connections
- LayerNorm
- GELU Feed-Forward Network
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
- [ ] MQA / GQA
- [ ] RoPE
- [ ] RMSNorm
- [ ] SwiGLU
- [ ] PyTorch SDPA / Flash Attention
- [ ] Mixture of Experts
- [ ] Scaling Law Experiments

## Project Structure

```text
Deerlight-GPT/
|-- deerlight_gpt.py      # First transparent single-file implementation
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

The training script automatically uses CUDA when a CUDA-enabled PyTorch build
is available, and otherwise falls back to CPU.

For the tested Windows + NVIDIA setup, install the CUDA-enabled PyTorch wheel
inside the Virtual Environment before running Training:

```powershell
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
```

## Current Milestone

Milestone 9 complete: cached Prefill/Decode is numerically checked against Full
Forward, and Cached/Uncached Generation has a controlled latency benchmark.

Next: implement and compare Multi-Query Attention and Grouped-Query Attention.
