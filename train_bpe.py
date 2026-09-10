"""Independent BPE training pilot; preserves all character-level checkpoints."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import time

import torch

from byte_bpe import ByteBPETokenizer
from deerlight_gpt import DeerlightGPTLanguageModel

ROOT = Path(__file__).resolve().parent


def prepare_data(tokenizer_dir):
    """Require provenance from evaluate_bpe.py and cache deterministic encoding."""
    text = (ROOT / "data/input.txt").read_text(encoding="utf-8")
    metadata = json.loads((tokenizer_dir / "metrics.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if metadata["text_sha256"] != digest:
        raise ValueError("Corpus differs from the Tokenizer experiment")
    split = metadata["split_character_index"]
    if split != int(len(text) * 0.9):
        raise ValueError("Expected the training-only 90/10 split")
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    tokenizer_json = tokenizer_path.read_text(encoding="utf-8")
    fingerprint = hashlib.sha256((digest + tokenizer_json + str(split)).encode()).hexdigest()
    cache_path = tokenizer_dir / f"encoded-{fingerprint}.pt"
    if cache_path.exists():
        data = torch.load(cache_path, weights_only=True)
    else:
        data = {}
        for name, sample in [("train", text[:split]), ("validation", text[split:])]:
            print(f"Encoding {name} (first run only)...", flush=True)
            ids = tokenizer.encode(sample)
            assert tokenizer.decode(ids) == sample
            data[name] = torch.tensor(ids, dtype=torch.long)
        torch.save(data, cache_path)
    return tokenizer, tokenizer_json, data, digest


def get_batch(data, batch_size, block_size, generator, device):
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    assert torch.equal(x[:, 1:], y[:, :-1])
    return x.to(device), y.to(device)


def load_bpe_checkpoint(path, device="cpu"):
    """Recover Tokenizer rules from the checkpoint, not from an unrelated file."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["format"] != "deerlight-bpe-model-v1":
        raise ValueError("Not a BPE checkpoint")
    payload = json.loads(checkpoint["tokenizer_json"])
    if (payload.get("format") != "deerlight-byte-bpe-v1"
            or payload.get("special_tokens") != {
                str(k): v for k, v in ByteBPETokenizer.SPECIAL_TOKENS.items()}):
        raise ValueError("Incompatible embedded Tokenizer")
    tokenizer = ByteBPETokenizer(payload["target_vocab_size"])
    for a, b, new_id in payload["merges"]:
        if (new_id != tokenizer.vocab_size or a not in tokenizer.vocab
                or b not in tokenizer.vocab or (a, b) in tokenizer.merges
                or new_id >= tokenizer.target_vocab_size):
            raise ValueError("Invalid embedded Tokenizer Rule")
        tokenizer.merges[(a, b)] = new_id
        tokenizer.vocab[new_id] = tokenizer.vocab[a] + tokenizer.vocab[b]
    if tokenizer.vocab_size != checkpoint["model_config"]["vocab_size"]:
        raise ValueError("Tokenizer/Model size mismatch")
    model = DeerlightGPTLanguageModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def main():
    parser = ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--eval-interval", type=int, default=250)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_interval < 1:
        parser.error("steps and eval-interval must be positive")
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, tokenizer_json, data, digest = prepare_data(args.tokenizer_dir)
    config = dict(vocab_size=tokenizer.vocab_size, embedding_dim=64,
                  num_heads=4, num_kv_heads=2, num_layers=2, block_size=64,
                  feed_forward_dim=176, attention_backend="manual")
    if any(len(tokens) <= config["block_size"] for tokens in data.values()):
        raise ValueError("Dataset too short")
    model = DeerlightGPTLanguageModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(1338)
    eval_generator = torch.Generator().manual_seed(1339)
    # Fixed held-out batches; evaluation never changes the training RNG stream.
    batches = [get_batch(data["validation"], 8, 64, eval_generator, device) for _ in range(5)]
    train_eval_generator = torch.Generator().manual_seed(1340)
    train_batches = [get_batch(data["train"], 8, 64, train_eval_generator, device) for _ in range(5)]
    output = ROOT / "runs" / ("bpe-model-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    history = []

    @torch.no_grad()
    def evaluate(evaluation_batches=batches):
        was_training = model.training
        model.eval()
        losses = [model(x, y)[1].item() for x, y in evaluation_batches]
        model.train(was_training)
        return sum(losses) / len(losses)

    initial_loss = evaluate()
    def save_checkpoint(path, step, validation_loss):
        torch.save(dict(format="deerlight-bpe-model-v1", model_config=config,
                        model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                        tokenizer_json=tokenizer_json, corpus_sha256=digest, step=step,
                        initial_validation_loss=initial_loss, validation_loss=validation_loss,
                        training_config=dict(batch_size=8, lr=3e-4, seed=1337,
                                             eval_interval=args.eval_interval, eval_batches=5)), path)

    best_loss, best_step = initial_loss, 0
    save_checkpoint(output / "best.pt", 0, initial_loss)
    history.append(dict(step=0, training_loss=evaluate(train_batches), validation_loss=initial_loss))
    print(f"Device: {device} | Vocabulary: {tokenizer.vocab_size} | Initial Validation Loss: {initial_loss:.4f}", flush=True)
    for step in range(1, args.steps + 1):
        x, y = get_batch(data["train"], 8, 64, generator, device)
        _, loss, _ = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.steps:
            final_loss = evaluate()
            training_loss = evaluate(train_batches)
            if final_loss < best_loss:
                best_loss, best_step = final_loss, step
                save_checkpoint(output / "best.pt", step, final_loss)
            history.append(dict(step=step, training_loss=training_loss, validation_loss=final_loss))
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            print(f"Step {step}: Train Eval {training_loss:.4f} | Validation {final_loss:.4f} | Best Step {best_step}", flush=True)
    path = output / "final.pt"
    save_checkpoint(path, args.steps, final_loss)
    restored, restored_tokenizer = load_bpe_checkpoint(path, device)
    model.eval()
    with torch.no_grad():
        expected = model(batches[0][0])[0]
        actual = restored(batches[0][0])[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored_tokenizer.encode("To be") == tokenizer.encode("To be")
    assert actual.shape == (8, 64, tokenizer.vocab_size)
    restored, restored_tokenizer = load_bpe_checkpoint(output / "best.pt", device)
    with torch.no_grad():
        reloaded_loss = sum(restored(x, y)[1].item() for x, y in batches) / len(batches)
    assert abs(reloaded_loss - best_loss) < 1e-6
    torch.manual_seed(1341)
    prompt = torch.tensor([restored_tokenizer.encode("To be")], device=device)
    generated = restored.generate(prompt, max_new_tokens=100, top_k=20)
    assert generated.shape[1] == prompt.shape[1] + 100
    preview = restored_tokenizer.decode(generated[0].tolist(), errors="replace", skip_special_tokens=True)
    (output / "sample.txt").write_text(preview, encoding="utf-8")
    print("Best Checkpoint sample:", ascii(preview))
    print(f"Best Validation Loss: {best_loss:.4f} at Step {best_step}")
    print(f"Final Validation Loss: {final_loss:.4f}")
    print("Shifted Batch, Forward/Backward, and checkpoint reload checks passed.")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
