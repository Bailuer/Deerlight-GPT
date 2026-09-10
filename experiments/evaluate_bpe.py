"""Train on the first 90% of raw text; evaluate byte compression on both splits."""

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from byte_bpe import ByteBPETokenizer


def main():
    text = (ROOT / "data" / "input.txt").read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    train_text, validation_text = text[:split], text[split:]
    if not train_text or not validation_text:
        raise ValueError("Both splits must be nonempty")
    output = ROOT / "runs" / ("bpe-512-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = ByteBPETokenizer(512)
    start = time.perf_counter()
    print("Training 253 Merge Rules on Training Split only...", flush=True)
    tokenizer.train(train_text)
    training_seconds = time.perf_counter() - start
    tokenizer.save(output / "tokenizer.json")
    restored = ByteBPETokenizer.load(output / "tokenizer.json")
    assert restored.merges == tokenizer.merges
    assert restored.vocab == tokenizer.vocab
    report = {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "split_character_index": split,
        "vocab_size": tokenizer.vocab_size,
        "merge_count": len(tokenizer.merges),
        "training_seconds": training_seconds,
        "special_tokens_added": False,
        "splits": {},
    }
    for name, sample in [("training", train_text), ("validation", validation_text)]:
        print(f"Encoding and checking {name}...", flush=True)
        ids = restored.encode(sample)
        assert restored.decode(ids) == sample, f"Round-trip failed: {name}"
        byte_count = len(sample.encode("utf-8"))
        metrics = {
            "characters": len(sample), "bytes": byte_count, "tokens": len(ids),
            "bytes_per_token": byte_count / len(ids), "round_trip": True,
        }
        report["splits"][name] = metrics
        print(name, metrics, flush=True)
    example = "To be, or not to be"
    ids = restored.encode(example)
    report["example"] = {"text": example, "ids": ids,
                         "pieces": [repr(restored.vocab[i]) for i in ids]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["example"], indent=2))
    print(f"Saved Tokenizer and metrics: {output}")


if __name__ == "__main__":
    main()
