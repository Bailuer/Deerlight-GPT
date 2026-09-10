"""Run a three-size educational IsoFLOP Scaling Law Pilot."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
import time

import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import deerlight_gpt as deerlight  # noqa: E402
import train_medium as medium  # noqa: E402


DEFAULT_TARGET_ND = 200_000_000_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_BLOCK_SIZE = 64
EVALUATION_BATCHES = 20
WARMUP_STEPS = 5

MODEL_SPECS = (
    {
        "name": "Tiny",
        "embedding_dim": 32,
        "num_heads": 2,
        "num_kv_heads": 1,
        "num_layers": 1,
    },
    {
        "name": "Mini",
        "embedding_dim": 48,
        "num_heads": 3,
        "num_kv_heads": 1,
        "num_layers": 2,
    },
    {
        "name": "Small",
        "embedding_dim": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 2,
    },
    {
        "name": "Medium",
        "embedding_dim": 96,
        "num_heads": 6,
        "num_kv_heads": 2,
        "num_layers": 3,
    },
    {
        "name": "Large",
        "embedding_dim": 128,
        "num_heads": 8,
        "num_kv_heads": 2,
        "num_layers": 4,
    },
)


def parse_arguments() -> Namespace:
    """Parse approximate Compute budget and common Batch configuration."""
    parser = ArgumentParser()
    parser.add_argument(
        "--target-nd",
        type=int,
        default=DEFAULT_TARGET_ND,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
    )
    parser.add_argument(
        "--attention-backend",
        choices=("manual", "sdpa", "sdpa_cudnn"),
        default=medium.MEDIUM_ATTENTION_BACKEND,
    )
    return parser.parse_args()


def build_config(
    spec: dict,
    vocab_size: int,
    block_size: int,
    attention_backend: str,
) -> dict:
    """Build one Dense Model configuration from a Scaling specification."""
    embedding_dim = spec["embedding_dim"]
    return {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "num_heads": spec["num_heads"],
        "num_kv_heads": spec["num_kv_heads"],
        "rope_base": deerlight.ROPE_BASE,
        "rms_norm_eps": deerlight.RMS_NORM_EPSILON,
        "num_layers": spec["num_layers"],
        "block_size": block_size,
        "feed_forward_dim": deerlight.calculate_swiglu_hidden_dim(
            embedding_dim=embedding_dim,
            multiple_of=deerlight.FEED_FORWARD_MULTIPLE_OF,
        ),
        "attention_backend": attention_backend,
    }


def parameter_count_for_config(config: dict) -> int:
    """Instantiate one CPU Model and return its exact Parameter count."""
    torch.manual_seed(deerlight.SEED)
    model = deerlight.DeerlightGPTLanguageModel(**config)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    del model
    return parameter_count


def make_fixed_batches(
    split: str,
    count: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    batch_size: int,
    block_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-generate one common Token stream for every Model configuration."""
    return [
        deerlight.get_batch(
            split=split,
            train_data=train_data,
            validation_data=validation_data,
            batch_size=batch_size,
            block_size=block_size,
        )
        for _ in range(count)
    ]


@torch.no_grad()
def evaluate(
    model: deerlight.DeerlightGPTLanguageModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Average Loss on identical fixed Validation Batches."""
    was_training = model.training
    model.eval()
    losses = []

    for x_batch, y_batch in batches:
        x_batch = x_batch.to(deerlight.DEVICE)
        y_batch = y_batch.to(deerlight.DEVICE)

        with medium.autocast_context():
            _, loss, _ = model(x_batch, y_batch)

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("IsoFLOP Evaluation produced non-finite Loss.")

        losses.append(loss.item())

    model.train(was_training)
    return sum(losses) / len(losses)


def train_configuration(
    name: str,
    config: dict,
    parameter_count: int,
    steps: int,
    training_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict:
    """Train one Model at its Compute-matched Token budget."""
    torch.manual_seed(deerlight.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(deerlight.SEED)

    model = deerlight.DeerlightGPTLanguageModel(
        **config,
    ).to(deerlight.DEVICE)
    optimizer_arguments = {"lr": medium.MEDIUM_LEARNING_RATE}

    if deerlight.DEVICE.type == "cuda":
        optimizer_arguments["fused"] = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        **optimizer_arguments,
    )
    initial_validation_loss = evaluate(
        model,
        validation_batches,
    )

    if deerlight.DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    medium.synchronize()
    wall_start = time.perf_counter()
    measured_seconds = 0.0
    measured_steps = 0
    final_training_loss = float("nan")

    for step, (x_batch, y_batch) in enumerate(
        training_batches[:steps],
        start=1,
    ):
        x_batch = x_batch.to(deerlight.DEVICE)
        y_batch = y_batch.to(deerlight.DEVICE)
        optimizer.zero_grad(set_to_none=True)

        if step > WARMUP_STEPS:
            medium.synchronize()
            step_start = time.perf_counter()

        with medium.autocast_context():
            _, loss, _ = model(x_batch, y_batch)

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced non-finite Training Loss.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=medium.MAX_GRADIENT_NORM,
        )
        optimizer.step()

        if step > WARMUP_STEPS:
            medium.synchronize()
            measured_seconds += time.perf_counter() - step_start
            measured_steps += 1

        final_training_loss = loss.item()

    medium.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    final_validation_loss = evaluate(
        model,
        validation_batches,
    )
    tokens_per_step = training_batches[0][0].numel()
    training_tokens = steps * tokens_per_step
    measured_tokens = measured_steps * tokens_per_step
    tokens_per_second = (
        measured_tokens / measured_seconds
        if measured_seconds > 0
        else 0.0
    )
    peak_allocated_gib = 0.0

    if deerlight.DEVICE.type == "cuda":
        peak_allocated_gib = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

    result = {
        "name": name,
        "embedding_dim": config["embedding_dim"],
        "num_layers": config["num_layers"],
        "parameters": parameter_count,
        "steps": steps,
        "training_tokens": training_tokens,
        "approximate_flops": 6 * parameter_count * training_tokens,
        "initial_validation_loss": initial_validation_loss,
        "final_training_loss": final_training_loss,
        "final_validation_loss": final_validation_loss,
        "tokens_per_second": tokens_per_second,
        "wall_seconds": wall_seconds,
        "peak_allocated_gib": peak_allocated_gib,
    }

    del optimizer
    del model

    if deerlight.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


def print_results(results: list[dict]) -> None:
    """Print the complete IsoFLOP Profile."""
    print(
        "| Model | C | Layers | Parameters | Steps | Training Tokens | "
        "Approx FLOPs | Tokens/s | Wall s | Peak GiB | Initial Val | "
        "Final Train | Final Val |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for result in results:
        print(
            f"| {result['name']} | "
            f"{result['embedding_dim']} | "
            f"{result['num_layers']} | "
            f"{result['parameters']:,} | "
            f"{result['steps']:,} | "
            f"{result['training_tokens']:,} | "
            f"{result['approximate_flops']:.3e} | "
            f"{result['tokens_per_second']:,.0f} | "
            f"{result['wall_seconds']:.2f} | "
            f"{result['peak_allocated_gib']:.3f} | "
            f"{result['initial_validation_loss']:.4f} | "
            f"{result['final_training_loss']:.4f} | "
            f"{result['final_validation_loss']:.4f} |"
        )


def main() -> None:
    """Run the fixed-Compute Model/Data allocation experiment."""
    args = parse_arguments()

    if args.target_nd <= 0:
        raise ValueError("Target N*D must be positive.")

    if args.batch_size <= 0 or args.block_size <= 0:
        raise ValueError("Batch Size and Block Size must be positive.")

    torch.manual_seed(deerlight.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(deerlight.SEED)
        torch.set_float32_matmul_precision("high")

    text = deerlight.load_text(deerlight.DATA_PATH)
    characters, stoi, _ = deerlight.build_vocabulary(text)
    data = deerlight.tokenize_dataset(text, stoi)
    train_data, validation_data = deerlight.split_dataset(
        data,
        deerlight.TRAIN_FRACTION,
    )
    configurations = []
    tokens_per_step = args.batch_size * args.block_size

    for spec in MODEL_SPECS:
        config = build_config(
            spec=spec,
            vocab_size=len(characters),
            block_size=args.block_size,
            attention_backend=args.attention_backend,
        )
        parameters = parameter_count_for_config(config)
        target_tokens = args.target_nd / parameters
        steps = max(1, round(target_tokens / tokens_per_step))
        configurations.append({
            "name": spec["name"],
            "config": config,
            "parameters": parameters,
            "steps": steps,
        })

    maximum_steps = max(
        configuration["steps"]
        for configuration in configurations
    )
    torch.manual_seed(deerlight.SEED + 1)
    training_batches = make_fixed_batches(
        split="train",
        count=maximum_steps,
        train_data=train_data,
        validation_data=validation_data,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )
    # Keep Validation Batches invariant when the Model search range changes.
    torch.manual_seed(deerlight.SEED + 2)
    validation_batches = make_fixed_batches(
        split="validation",
        count=EVALUATION_BATCHES,
        train_data=train_data,
        validation_data=validation_data,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )

    print("Deerlight GPT Mini IsoFLOP Scaling Pilot")
    print("Device:", deerlight.DEVICE)

    if deerlight.DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(deerlight.DEVICE))

    print("Target N*D:", f"{args.target_nd:.3e}")
    print("Batch Shape:", (args.batch_size, args.block_size))
    print("Tokenizer: Character-level")
    print("Underlying Training Tokens:", len(train_data))

    results = [
        train_configuration(
            name=configuration["name"],
            config=configuration["config"],
            parameter_count=configuration["parameters"],
            steps=configuration["steps"],
            training_batches=training_batches,
            validation_batches=validation_batches,
        )
        for configuration in configurations
    ]
    print_results(results)


if __name__ == "__main__":
    main()
