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

## Roadmap

### Baseline Deerlight GPT

- [ ] Character-level Tokenizer
- [ ] Training and Validation Split
- [ ] Shifted Training Batches
- [ ] Bigram Language Model
- [ ] Single-Head Self-Attention
- [ ] Multi-Head Self-Attention
- [ ] Transformer Block
- [ ] Decoder-only GPT
- [ ] Training Loop
- [ ] Autoregressive Generation

### Modern LLM Upgrades

- [ ] KV Cache
- [ ] RoPE
- [ ] RMSNorm
- [ ] SwiGLU
- [ ] MQA / GQA
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

## Current Milestone

Milestone 1: implement and verify the Character-level Tokenizer and shifted Batch Data Pipeline.

