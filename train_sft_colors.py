"""Tiny author-created color translation SFT; not a general chat benchmark."""

from argparse import ArgumentParser
import json
from pathlib import Path
import time

import torch

from sft_data import SFTDataCollator, encode_prompt
from train_bpe import ROOT, load_bpe_checkpoint


def make_dataset():
    colors = {"red": "红色", "blue": "蓝色", "green": "绿色", "black": "黑色"}
    # All colors are seen; only the wording template is held out.
    train_templates = ["Translate {c} into Chinese.", "Chinese for {c}?"]
    validation_templates = ["What is {c} in Chinese?"]
    return {split: [{"user": template.format(c=color), "assistant": answer}
                    for template in templates for color, answer in colors.items()]
            for split, templates in [("train", train_templates), ("validation", validation_templates)]}


@torch.no_grad()
def generate_answer(model, tokenizer, user, device, max_new_tokens=16):
    """Greedy continuation from the prompt ONLY; EOS ends the loop."""
    ids = encode_prompt(tokenizer, user)
    generated = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.block_size:]], device=device)
        logits, _, _ = model(x)
        next_id = int(logits[0, -1].argmax())
        if next_id == tokenizer.EOS_ID:
            return tokenizer.decode(generated, errors="replace"), True
        generated.append(next_id)
        ids.append(next_id)
    return tokenizer.decode(generated, errors="replace"), False


@torch.no_grad()
def evaluate(model, tokenizer, dataset, collator, device):
    was_training = model.training
    model.eval()
    result = {}
    try:
        for split, examples in dataset.items():
            batch = {k: v.to(device) for k, v in collator(examples).items()}
            loss = model(batch["input_ids"], batch["labels"])[1].item()
            rows = []
            for item in examples:
                answer, stopped = generate_answer(model, tokenizer, item["user"], device)
                rows.append(dict(**item, prediction=answer, eos=stopped,
                                 exact_match=answer == item["assistant"]))
            result[split] = dict(loss=loss, exact_match=sum(r["exact_match"] for r in rows)/len(rows),
                                 eos_rate=sum(r["eos"] for r in rows)/len(rows), rows=rows)
    finally:
        model.train(was_training)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.manual_seed(1342)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, tokenizer = load_bpe_checkpoint(args.checkpoint, device)
    dataset = make_dataset()
    collator = SFTDataCollator(tokenizer, model.block_size)
    batch = {k: v.to(device) for k, v in collator(dataset["train"]).items()}
    assert set(x["user"] for x in dataset["train"]).isdisjoint(x["user"] for x in dataset["validation"])
    output = ROOT / "runs" / ("sft-colors-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    best_loss = float("inf")
    best_step = 0
    for step in range(args.steps + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch["input_ids"], batch["labels"])[1]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        if step % 200 == 0 or step == args.steps:
            metrics = evaluate(model, tokenizer, dataset, collator, device)
            history.append(dict(step=step, **metrics))
            if metrics["validation"]["loss"] < best_loss:
                best_loss = metrics["validation"]["loss"]
                best_step = step
                checkpoint = dict(format="deerlight-bpe-model-v1", model_config=source["model_config"],
                    model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    tokenizer_json=source["tokenizer_json"], corpus_sha256=source["corpus_sha256"],
                    step=step, stage="sft-colors", source_checkpoint=str(args.checkpoint.resolve()),
                    validation_loss=best_loss, dataset=dataset,
                    training_config=dict(seed=1342, lr=3e-4, steps=args.steps,
                        chat_format="sft_data.encode_prompt-v1; separate prompt/answer encoding",
                        batch_size=len(dataset["train"]), generation="greedy; EOS or 16 tokens"))
                torch.save(checkpoint, output / "best.pt")
            print(f"Step {step} | Train Loss {metrics['train']['loss']:.4f} | Val Loss {metrics['validation']['loss']:.4f} | Train EM {metrics['train']['exact_match']:.0%} | Val EM {metrics['validation']['exact_match']:.0%} | Val EOS {metrics['validation']['eos_rate']:.0%}", flush=True)
            (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    restored, restored_tokenizer = load_bpe_checkpoint(output / "best.pt", device)
    best_metrics = evaluate(restored, restored_tokenizer, dataset, collator, device)
    assert abs(best_metrics["validation"]["loss"] - best_loss) < 1e-6
    selected = next(row for row in history if row["step"] == best_step)
    assert best_metrics["validation"]["rows"] == selected["validation"]["rows"]
    (output / "best_metrics.json").write_text(json.dumps(dict(step=best_step, **best_metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Best step:", best_step)
    for row in best_metrics["validation"]["rows"]:
        print(ascii(row))
    print("Checkpoint reload verified. Saved:", output)


if __name__ == "__main__":
    main()
