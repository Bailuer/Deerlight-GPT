"""Byte-normalized canonical BPE NLL with explicit block-reset context policy.

Each block scores up to T targets using 1..T preceding tokens. The next block
retains only the previous target as its first input: no cross-block KV cache.
Every token except token 0 is a target exactly once. This is NOT a full sliding
window evaluation, and fixed token contexts are not equal byte contexts across
different Tokenizers.
"""

from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from train_bpe import ROOT, load_bpe_checkpoint


def target_blocks(ids, block_size):
    if block_size < 1:
        raise ValueError("block_size must be positive")
    for start in range(0, len(ids) - 1, block_size):
        end = min(start + block_size, len(ids) - 1)
        yield ids[start:end], ids[start + 1:end + 1]


@torch.no_grad()
def score_bpb(model, tokenizer, ids, block_size, device="cpu"):
    if len(ids) < 2:
        raise ValueError("At least two Tokens are required")
    if any(i not in tokenizer.vocab for i in ids):
        raise ValueError("Only ordinary Byte/BPE Tokens are supported")
    target_bytes = sum(len(tokenizer.vocab[i]) for i in ids[1:])
    if target_bytes == 0:
        raise ValueError("No target bytes")
    was_training = model.training
    model.eval()
    total_nll, target_count = 0.0, 0
    try:
        for inputs, targets in target_blocks(ids, block_size):
            x = torch.tensor([inputs], dtype=torch.long, device=device)
            y = torch.tensor(targets, dtype=torch.long, device=device)
            logits, _, _ = model(x)
            nll = F.cross_entropy(logits[0].float(), y, reduction="none")
            if not torch.isfinite(nll).all():
                raise ValueError("Non-finite NLL")
            total_nll += nll.double().sum().item()
            target_count += len(targets)
    finally:
        model.train(was_training)
    assert target_count == len(ids) - 1
    return dict(total_nll_nats=total_nll, target_tokens=target_count,
                target_bytes=target_bytes, excluded_prefix_bytes=len(tokenizer.vocab[ids[0]]),
                mean_token_nll=total_nll / target_count,
                bpb=total_nll / target_bytes / math.log(2),
                uniform_bpb=target_count * math.log2(tokenizer.vocab_size) / target_bytes,
                context_policy="block-reset; 1..T preceding Tokens; no KV reuse",
                block_size=block_size)


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    text = (ROOT / "data/input.txt").read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != checkpoint["corpus_sha256"]:
        raise ValueError("Corpus does not match Checkpoint")
    validation = text[int(len(text) * 0.9):]
    model, tokenizer = load_bpe_checkpoint(args.checkpoint, device)
    ids = tokenizer.encode(validation)
    assert tokenizer.decode(ids) == validation
    report = score_bpb(model, tokenizer, ids, model.block_size, device)
    assert report["target_bytes"] + report["excluded_prefix_bytes"] == len(validation.encode("utf-8"))
    report.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_step=checkpoint["step"],
                  validation_sha256=hashlib.sha256(validation.encode("utf-8")).hexdigest(),
                  tokenizer_sha256=hashlib.sha256(checkpoint["tokenizer_json"].encode()).hexdigest(),
                  split_policy="last 10% by character index", device=device,
                  note="Validation, not independent Test. Canonical encoding NLL; not a sum over alternate tokenizations.")
    output = ROOT / "runs" / ("bpb-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
