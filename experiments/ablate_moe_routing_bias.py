"""Ablate loss-free MoE Routing Bias with all other variables fixed."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))

import benchmark_moe as benchmark  # noqa: E402
import deerlight_gpt as deerlight  # noqa: E402
import train_medium as medium  # noqa: E402


DEFAULT_STEPS = 100
DEFAULT_BATCH_SIZE = 8
CONTROL_RATE = 0.0
TREATMENT_RATE = 1e-3


def parse_arguments() -> Namespace:
    """Parse Ablation length, Batch size, and Attention backend."""
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--attention-backend",
        choices=("manual", "sdpa", "sdpa_cudnn"),
        default=medium.MEDIUM_ATTENTION_BACKEND,
    )
    return parser.parse_args()


def average_layer_metric(
    result: dict,
    metric_name: str,
) -> float:
    """Average one load metric across all MoE Layers."""
    metrics = result["load_metrics"].values()
    return sum(
        layer_metrics[metric_name]
        for layer_metrics in metrics
    ) / len(result["load_metrics"])


def print_ablation_summary(results: list[dict]) -> None:
    """Print the load-balance metrics that isolate Routing Bias behavior."""
    print(
        "| Routing Bias Rate | Early Mean CV | Late Mean CV | "
        "Cumulative CV | Zero-Selection Rate | Final Val Loss |"
    )
    print("|---:|---:|---:|---:|---:|---:|")

    for rate, result in zip((CONTROL_RATE, TREATMENT_RATE), results):
        print(
            f"| {rate:.3f} | "
            f"{average_layer_metric(result, 'early_mean_cv'):.4f} | "
            f"{average_layer_metric(result, 'late_mean_cv'):.4f} | "
            f"{average_layer_metric(result, 'coefficient_of_variation'):.4f} | "
            f"{average_layer_metric(result, 'zero_selection_rate'):.3%} | "
            f"{result['final_validation_loss']:.4f} |"
        )


def main() -> None:
    """Run otherwise-identical MoE Pilots with Bias updates off and on."""
    args = parse_arguments()

    if args.steps <= benchmark.WARMUP_STEPS:
        raise ValueError(
            f"Steps must be greater than {benchmark.WARMUP_STEPS}."
        )

    if args.batch_size <= 0:
        raise ValueError("Batch Size must be positive.")

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

    torch.manual_seed(deerlight.SEED + 1)
    training_batches = benchmark.make_fixed_batches(
        split="train",
        count=args.steps,
        train_data=train_data,
        validation_data=validation_data,
        batch_size=args.batch_size,
    )
    evaluation_batches = benchmark.make_fixed_batches(
        split="validation",
        count=benchmark.EVALUATION_BATCHES,
        train_data=train_data,
        validation_data=validation_data,
        batch_size=args.batch_size,
    )
    dense_config = benchmark.build_config(
        vocab_size=len(characters),
        attention_backend=args.attention_backend,
        use_moe=False,
    )
    control_config = benchmark.build_config(
        vocab_size=len(characters),
        attention_backend=args.attention_backend,
        use_moe=True,
    )
    treatment_config = dict(control_config)
    control_config["moe_routing_bias_update_rate"] = CONTROL_RATE
    treatment_config["moe_routing_bias_update_rate"] = TREATMENT_RATE

    torch.manual_seed(deerlight.SEED)
    reference_model = deerlight.DeerlightGPTLanguageModel(
        **dense_config,
    )
    reference_state = {
        name: tensor.detach().clone()
        for name, tensor in reference_model.state_dict().items()
    }
    del reference_model

    print("Loss-Free Routing Bias Ablation")
    print("Device:", deerlight.DEVICE)

    if deerlight.DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(deerlight.DEVICE))

    print("Steps:", args.steps)
    print("Batch Shape:", (args.batch_size, medium.MEDIUM_BLOCK_SIZE))
    print("Control Rate:", CONTROL_RATE)
    print("Treatment Rate:", TREATMENT_RATE)

    results = [
        benchmark.train_configuration(
            name="MoE Bias Off",
            config=control_config,
            reference_state=reference_state,
            training_batches=training_batches,
            evaluation_batches=evaluation_batches,
        ),
        benchmark.train_configuration(
            name="MoE Bias 0.001",
            config=treatment_config,
            reference_state=reference_state,
            training_batches=training_batches,
            evaluation_batches=evaluation_batches,
        ),
    ]
    benchmark.print_results(results)
    print("\nFocused Bias Ablation:")
    print_ablation_summary(results)


if __name__ == "__main__":
    main()
