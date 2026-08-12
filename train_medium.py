"""Train the modern Deerlight GPT Medium capability baseline."""

from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from pathlib import Path
import time

import torch

import deerlight_gpt as deerlight


MEDIUM_EMBEDDING_DIM = 384
MEDIUM_NUM_QUERY_HEADS = 6
MEDIUM_NUM_KV_HEADS = 2
MEDIUM_NUM_LAYERS = 6
MEDIUM_BLOCK_SIZE = 256
MEDIUM_BATCH_SIZE = 32
MEDIUM_LEARNING_RATE = 3e-4
MAX_GRADIENT_NORM = 1.0
MEDIUM_TRAINING_STEPS = 5_000
MEDIUM_EVAL_INTERVAL = 250
MEDIUM_EVAL_BATCHES = 20
PILOT_STEPS = 30
PILOT_EVAL_BATCHES = 5
PILOT_PRINT_INTERVAL = 5
TRAIN_PRINT_INTERVAL = 50
EARLY_STOPPING_PATIENCE = 4
GENERATION_LENGTH = 300

MEDIUM_CHECKPOINT_PATH = (
    deerlight.PROJECT_DIR
    / "checkpoints"
    / "deerlight_gpt_medium_best.pt"
)
PILOT_CHECKPOINT_PATH = (
    deerlight.PROJECT_DIR
    / "checkpoints"
    / "deerlight_gpt_medium_pilot_best.pt"
)


def autocast_context():
    """Use BF16 Activations on CUDA and full FP32 on CPU."""
    if deerlight.DEVICE.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    return nullcontext()


def synchronize() -> None:
    """Wait for asynchronous CUDA work before reading Wall-clock time."""
    if deerlight.DEVICE.type == "cuda":
        torch.cuda.synchronize()


def build_model_config(vocab_size: int) -> dict[str, int | float]:
    """Return the complete reproducible Medium Architecture configuration."""
    feed_forward_dim = deerlight.calculate_swiglu_hidden_dim(
        embedding_dim=MEDIUM_EMBEDDING_DIM,
        multiple_of=deerlight.FEED_FORWARD_MULTIPLE_OF,
    )

    return {
        "vocab_size": vocab_size,
        "embedding_dim": MEDIUM_EMBEDDING_DIM,
        "num_heads": MEDIUM_NUM_QUERY_HEADS,
        "num_kv_heads": MEDIUM_NUM_KV_HEADS,
        "rope_base": deerlight.ROPE_BASE,
        "rms_norm_eps": deerlight.RMS_NORM_EPSILON,
        "num_layers": MEDIUM_NUM_LAYERS,
        "block_size": MEDIUM_BLOCK_SIZE,
        "feed_forward_dim": feed_forward_dim,
    }


@torch.no_grad()
def estimate_losses(
    model: deerlight.DeerlightGPTLanguageModel,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    batch_size: int,
    evaluation_batches: int,
) -> dict[str, float]:
    """Estimate Train and Validation Loss with the selected precision."""
    was_training = model.training
    model.eval()
    losses = {}

    for split in ("train", "validation"):
        batch_losses = []

        for _ in range(evaluation_batches):
            x_batch, y_batch = deerlight.get_batch(
                split=split,
                train_data=train_data,
                validation_data=validation_data,
                batch_size=batch_size,
                block_size=MEDIUM_BLOCK_SIZE,
            )
            x_batch = x_batch.to(deerlight.DEVICE)
            y_batch = y_batch.to(deerlight.DEVICE)

            with autocast_context():
                _, loss, _ = model(x_batch, y_batch)

            if loss is None:
                raise RuntimeError("Medium Evaluation Loss is None.")

            batch_losses.append(loss.item())

        losses[split] = sum(batch_losses) / len(batch_losses)

    model.train(was_training)
    return losses


def parse_arguments() -> Namespace:
    """Parse Pilot or full Training options."""
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("pilot", "train"),
        default="pilot",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=MEDIUM_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    """Run a short resource Pilot or the full Medium Training experiment."""
    args = parse_arguments()

    if args.batch_size <= 0:
        raise ValueError("Batch Size must be positive.")

    default_steps = (
        PILOT_STEPS
        if args.mode == "pilot"
        else MEDIUM_TRAINING_STEPS
    )
    training_steps = args.steps or default_steps

    if training_steps <= 0:
        raise ValueError("Training Steps must be positive.")

    evaluation_batches = (
        PILOT_EVAL_BATCHES
        if args.mode == "pilot"
        else MEDIUM_EVAL_BATCHES
    )
    evaluation_interval = (
        training_steps
        if args.mode == "pilot"
        else MEDIUM_EVAL_INTERVAL
    )
    checkpoint_path = (
        PILOT_CHECKPOINT_PATH
        if args.mode == "pilot"
        else MEDIUM_CHECKPOINT_PATH
    )
    print_interval = (
        PILOT_PRINT_INTERVAL
        if args.mode == "pilot"
        else TRAIN_PRINT_INTERVAL
    )

    torch.manual_seed(deerlight.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(deerlight.SEED)
        torch.set_float32_matmul_precision("high")

    text = deerlight.load_text(deerlight.DATA_PATH)
    characters, stoi, itos = deerlight.build_vocabulary(text)
    data = deerlight.tokenize_dataset(text, stoi)
    train_data, validation_data = deerlight.split_dataset(
        data,
        deerlight.TRAIN_FRACTION,
    )
    model_config = build_model_config(len(characters))
    model = deerlight.DeerlightGPTLanguageModel(
        **model_config,
    ).to(deerlight.DEVICE)

    optimizer_arguments = {
        "lr": MEDIUM_LEARNING_RATE,
    }

    if deerlight.DEVICE.type == "cuda":
        optimizer_arguments["fused"] = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        **optimizer_arguments,
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Deerlight GPT Medium")
    print("Mode:", args.mode)
    print("Device:", deerlight.DEVICE)

    if deerlight.DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(deerlight.DEVICE))

    print("Parameters:", f"{parameter_count:,}")
    print("Model Config:", model_config)
    print("Batch Shape:", (args.batch_size, MEDIUM_BLOCK_SIZE))
    print("Tokens per Step:", args.batch_size * MEDIUM_BLOCK_SIZE)
    print("Precision:", "BF16 Autocast" if torch.cuda.is_available() else "FP32")

    initial_losses = estimate_losses(
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        batch_size=args.batch_size,
        evaluation_batches=evaluation_batches,
    )
    print(
        f"Step 0/{training_steps} | "
        f"Train Loss: {initial_losses['train']:.4f} | "
        f"Validation Loss: {initial_losses['validation']:.4f}"
    )

    best_validation_loss = float("inf")
    evaluations_without_improvement = 0
    warmup_steps = min(5, max(0, training_steps // 5))
    measured_training_seconds = 0.0
    measured_training_steps = 0

    if deerlight.DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()

    for step in range(1, training_steps + 1):
        x_batch, y_batch = deerlight.get_batch(
            split="train",
            train_data=train_data,
            validation_data=validation_data,
            batch_size=args.batch_size,
            block_size=MEDIUM_BLOCK_SIZE,
        )
        x_batch = x_batch.to(deerlight.DEVICE)
        y_batch = y_batch.to(deerlight.DEVICE)

        if step > warmup_steps:
            synchronize()
            step_start = time.perf_counter()

        with autocast_context():
            _, training_loss, _ = model(x_batch, y_batch)

        if training_loss is None or not torch.isfinite(training_loss):
            raise RuntimeError("Medium Training produced a non-finite Loss.")

        optimizer.zero_grad(set_to_none=True)
        training_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRADIENT_NORM,
        )
        optimizer.step()

        if step > warmup_steps:
            synchronize()
            measured_training_seconds += time.perf_counter() - step_start
            measured_training_steps += 1

        if step % print_interval == 0 or step == 1:
            print(
                f"Step {step:4d}/{training_steps} | "
                f"Batch Loss: {training_loss.item():.4f} | "
                f"Gradient Norm: {gradient_norm.item():.4f}"
            )

        should_evaluate = (
            step % evaluation_interval == 0
            or step == training_steps
        )

        if should_evaluate:
            losses = estimate_losses(
                model=model,
                train_data=train_data,
                validation_data=validation_data,
                batch_size=args.batch_size,
                evaluation_batches=evaluation_batches,
            )
            print(
                f"Evaluation {step:4d}/{training_steps} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Validation Loss: {losses['validation']:.4f}"
            )

            if losses["validation"] < best_validation_loss:
                best_validation_loss = losses["validation"]
                evaluations_without_improvement = 0
                deerlight.save_deerlight_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    training_loss=losses["train"],
                    validation_loss=losses["validation"],
                    model_config=model_config,
                    characters=characters,
                )
                print("Saved Best Checkpoint:", checkpoint_path.name)
            else:
                evaluations_without_improvement += 1

            if (
                args.mode == "train"
                and evaluations_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                print(
                    "Early Stopping: Validation Loss did not improve for",
                    EARLY_STOPPING_PATIENCE,
                    "evaluations.",
                )
                break

    measured_tokens = (
        measured_training_steps
        * args.batch_size
        * MEDIUM_BLOCK_SIZE
    )
    tokens_per_second = (
        measured_tokens / measured_training_seconds
        if measured_training_seconds > 0
        else 0.0
    )

    print("\nPilot/Training Metrics:")
    print("Measured Steps:", measured_training_steps)
    print("Training Seconds:", f"{measured_training_seconds:.3f}")
    print("Tokens per Second:", f"{tokens_per_second:,.0f}")
    print("Best Validation Loss:", f"{best_validation_loss:.4f}")

    if deerlight.DEVICE.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print("Peak VRAM Allocated GiB:", f"{peak_allocated:.3f}")
        print("Peak VRAM Reserved GiB:", f"{peak_reserved:.3f}")

    best_model, best_checkpoint = deerlight.load_deerlight_checkpoint(
        checkpoint_path=checkpoint_path,
        device=deerlight.DEVICE,
    )
    print("Best Checkpoint Step:", best_checkpoint["step"])
    print(
        "Best Checkpoint Validation Loss:",
        f"{best_checkpoint['validation_loss']:.4f}",
    )

    prompt_token = stoi.get("\n", 0)
    generated = best_model.generate(
        torch.tensor(
            [[prompt_token]],
            dtype=torch.long,
            device=deerlight.DEVICE,
        ),
        max_new_tokens=GENERATION_LENGTH,
        temperature=deerlight.GENERATION_TEMPERATURE,
        top_k=deerlight.GENERATION_TOP_K,
    )
    print("\nGenerated Sample:")
    print(
        deerlight.decode(
            generated[0].detach().cpu().tolist(),
            itos,
        )
    )


if __name__ == "__main__":
    main()
