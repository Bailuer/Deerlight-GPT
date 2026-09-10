"""Compare Dense and Interleaved Sparse-MoE Deerlight GPT training Pilots."""

from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import deerlight_gpt as deerlight  # noqa: E402
import train_medium as medium  # noqa: E402


MOE_LAYER_INDICES = [1, 3, 5]
MOE_NUM_EXPERTS = 8
MOE_TOP_K = 2
MOE_EXPERT_HIDDEN_DIM = 352
MOE_ROUTING_BIAS_UPDATE_RATE = 1e-3
DEFAULT_STEPS = 30
DEFAULT_BATCH_SIZE = 8
EVALUATION_BATCHES = 5
WARMUP_STEPS = 5
LOAD_MOVING_AVERAGE_WINDOW = 10


def parse_arguments() -> Namespace:
    """Parse Pilot length, Batch size, and Attention backend."""
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


def autocast_context():
    """Use the same Training precision as the Medium baseline."""
    if deerlight.DEVICE.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    return nullcontext()


def synchronize() -> None:
    """Wait for asynchronous CUDA work before reading the Timer."""
    if deerlight.DEVICE.type == "cuda":
        torch.cuda.synchronize()


def build_config(
    vocab_size: int,
    attention_backend: str,
    use_moe: bool,
) -> dict:
    """Return one Dense or matched-active-capacity MoE configuration."""
    config = medium.build_model_config(
        vocab_size=vocab_size,
        attention_backend=attention_backend,
    )

    if use_moe:
        config.update({
            "moe_layer_indices": MOE_LAYER_INDICES,
            "moe_expert_hidden_dim": MOE_EXPERT_HIDDEN_DIM,
            "moe_num_experts": MOE_NUM_EXPERTS,
            "moe_top_k": MOE_TOP_K,
            "moe_routing_bias_update_rate": (
                MOE_ROUTING_BIAS_UPDATE_RATE
            ),
        })

    return config


def make_fixed_batches(
    split: str,
    count: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-generate CPU Batches so both Models see identical Tokens."""
    return [
        deerlight.get_batch(
            split=split,
            train_data=train_data,
            validation_data=validation_data,
            batch_size=batch_size,
            block_size=medium.MEDIUM_BLOCK_SIZE,
        )
        for _ in range(count)
    ]


def copy_compatible_state(
    model: deerlight.DeerlightGPTLanguageModel,
    reference_state: dict[str, torch.Tensor],
) -> int:
    """Copy every same-name, same-Shape Tensor from the Dense reference."""
    copied_tensors = 0

    with torch.no_grad():
        for name, target in model.state_dict().items():
            source = reference_state.get(name)

            if source is None or source.shape != target.shape:
                continue

            target.copy_(source.to(target.device))
            copied_tensors += 1

    return copied_tensors


def per_token_active_parameter_count(
    model: deerlight.DeerlightGPTLanguageModel,
) -> int:
    """Estimate Parameters touched by one Token's selected Expert paths."""
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    inactive_expert_parameters = 0

    for module in model.modules():
        if not isinstance(module, deerlight.SparseMoE):
            continue

        one_expert_parameters = sum(
            parameter.numel()
            for parameter in module.routed_experts[0].parameters()
        )
        inactive_expert_parameters += (
            module.num_experts - module.top_k
        ) * one_expert_parameters

    return total_parameters - inactive_expert_parameters


@torch.no_grad()
def evaluate_fixed_batches(
    model: deerlight.DeerlightGPTLanguageModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Return Mean Loss over fixed Batches without updating Routing Bias."""
    was_training = model.training
    model.eval()
    losses = []

    for x_batch, y_batch in batches:
        x_batch = x_batch.to(deerlight.DEVICE)
        y_batch = y_batch.to(deerlight.DEVICE)

        with autocast_context():
            _, loss, _ = model(x_batch, y_batch)

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Evaluation produced a non-finite Loss.")

        losses.append(loss.item())

    model.train(was_training)
    return sum(losses) / len(losses)


def collect_moe_modules(
    model: deerlight.DeerlightGPTLanguageModel,
) -> dict[int, deerlight.SparseMoE]:
    """Return MoE modules keyed by zero-based Transformer Layer index."""
    modules = {}

    for layer_index, block in enumerate(model.transformer_blocks):
        if isinstance(block.feed_forward, deerlight.SparseMoE):
            modules[layer_index] = block.feed_forward

    return modules


def train_configuration(
    name: str,
    config: dict,
    reference_state: dict[str, torch.Tensor],
    training_batches: list[tuple[torch.Tensor, torch.Tensor]],
    evaluation_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict:
    """Train one configuration and return comparable Pilot metrics."""
    torch.manual_seed(deerlight.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(deerlight.SEED)

    model = deerlight.DeerlightGPTLanguageModel(
        **config,
    ).to(deerlight.DEVICE)
    copied_tensors = copy_compatible_state(
        model,
        reference_state,
    )
    optimizer_arguments = {"lr": medium.MEDIUM_LEARNING_RATE}

    if deerlight.DEVICE.type == "cuda":
        optimizer_arguments["fused"] = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        **optimizer_arguments,
    )
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    active_parameters = per_token_active_parameter_count(model)
    moe_modules = collect_moe_modules(model)
    cumulative_loads = {
        layer_index: torch.zeros(
            module.num_experts,
            dtype=torch.long,
        )
        for layer_index, module in moe_modules.items()
    }
    first_loads = {}
    last_loads = {}
    load_cv_history = {
        layer_index: []
        for layer_index in moe_modules
    }
    zero_selection_counts = {
        layer_index: 0
        for layer_index in moe_modules
    }
    initial_validation_loss = evaluate_fixed_batches(
        model,
        evaluation_batches,
    )

    if deerlight.DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    measured_seconds = 0.0
    measured_steps = 0
    final_training_loss = float("nan")
    model.train()

    for step, (x_batch, y_batch) in enumerate(training_batches, start=1):
        x_batch = x_batch.to(deerlight.DEVICE)
        y_batch = y_batch.to(deerlight.DEVICE)
        optimizer.zero_grad(set_to_none=True)

        if step > WARMUP_STEPS:
            synchronize()
            step_start = time.perf_counter()

        with autocast_context():
            _, loss, _ = model(x_batch, y_batch)

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced a non-finite Training Loss.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=medium.MAX_GRADIENT_NORM,
        )
        optimizer.step()

        if step > WARMUP_STEPS:
            synchronize()
            measured_seconds += time.perf_counter() - step_start
            measured_steps += 1

        for layer_index, module in moe_modules.items():
            current_loads = module.last_load_counts.detach().cpu().clone()
            cumulative_loads[layer_index] += current_loads

            if step == 1:
                first_loads[layer_index] = current_loads

            last_loads[layer_index] = current_loads
            current_loads_float = current_loads.float()
            load_cv_history[layer_index].append(
                (
                    current_loads_float.std(unbiased=False)
                    / current_loads_float.mean()
                ).item()
            )
            zero_selection_counts[layer_index] += (
                current_loads == 0
            ).sum().item()

        final_training_loss = loss.item()

    final_validation_loss = evaluate_fixed_batches(
        model,
        evaluation_batches,
    )
    measured_tokens = (
        measured_steps
        * training_batches[0][0].numel()
    )
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

    load_metrics = {}

    for layer_index, loads in cumulative_loads.items():
        expected_assignments = (
            len(training_batches)
            * training_batches[0][0].numel()
            * moe_modules[layer_index].top_k
        )

        if loads.sum().item() != expected_assignments:
            raise RuntimeError(
                f"Layer {layer_index} recorded {loads.sum().item()} Routed "
                f"assignments; expected {expected_assignments}."
            )

        loads_float = loads.float()
        mean_load = loads_float.mean()
        first_loads_float = first_loads[layer_index].float()
        last_loads_float = last_loads[layer_index].float()
        cv_history = load_cv_history[layer_index]
        window = min(
            LOAD_MOVING_AVERAGE_WINDOW,
            len(cv_history),
        )
        load_metrics[layer_index] = {
            "counts": loads.tolist(),
            "first_counts": first_loads[layer_index].tolist(),
            "last_counts": last_loads[layer_index].tolist(),
            "max_over_mean": (
                loads_float.max() / mean_load
            ).item(),
            "coefficient_of_variation": (
                loads_float.std(unbiased=False) / mean_load
            ).item(),
            "first_coefficient_of_variation": (
                first_loads_float.std(unbiased=False)
                / first_loads_float.mean()
            ).item(),
            "last_coefficient_of_variation": (
                last_loads_float.std(unbiased=False)
                / last_loads_float.mean()
            ).item(),
            "early_mean_cv": sum(cv_history[:window]) / window,
            "late_mean_cv": sum(cv_history[-window:]) / window,
            "zero_selection_rate": (
                zero_selection_counts[layer_index]
                / (
                    len(training_batches)
                    * moe_modules[layer_index].num_experts
                )
            ),
            "routing_bias": (
                moe_modules[layer_index]
                .routing_bias.detach().cpu().tolist()
            ),
        }

    result = {
        "name": name,
        "total_parameters": total_parameters,
        "active_parameters": active_parameters,
        "copied_tensors": copied_tensors,
        "initial_validation_loss": initial_validation_loss,
        "final_training_loss": final_training_loss,
        "final_validation_loss": final_validation_loss,
        "tokens_per_second": tokens_per_second,
        "peak_allocated_gib": peak_allocated_gib,
        "load_metrics": load_metrics,
    }

    del optimizer
    del model

    if deerlight.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


def print_results(results: list[dict]) -> None:
    """Print comparable summary and per-layer MoE load diagnostics."""
    print(
        "| Model | Total Params | Active Params/Token | Tokens/s | "
        "Peak VRAM GiB | Initial Val Loss | Final Train Loss | "
        "Final Val Loss |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for result in results:
        print(
            f"| {result['name']} | "
            f"{result['total_parameters']:,} | "
            f"{result['active_parameters']:,} | "
            f"{result['tokens_per_second']:,.0f} | "
            f"{result['peak_allocated_gib']:.3f} | "
            f"{result['initial_validation_loss']:.4f} | "
            f"{result['final_training_loss']:.4f} | "
            f"{result['final_validation_loss']:.4f} |"
        )

        for layer_index, metrics in result["load_metrics"].items():
            rounded_bias = [
                round(value, 4)
                for value in metrics["routing_bias"]
            ]
            print(
                f"  Layer {layer_index} Cumulative Loads: "
                f"{metrics['counts']} | "
                f"Max/Mean: {metrics['max_over_mean']:.3f} | "
                f"CV: {metrics['coefficient_of_variation']:.3f} | "
                f"Bias: {rounded_bias}"
            )
            print(
                f"    First -> Last Loads: {metrics['first_counts']} -> "
                f"{metrics['last_counts']} | CV: "
                f"{metrics['first_coefficient_of_variation']:.3f} -> "
                f"{metrics['last_coefficient_of_variation']:.3f}"
            )
            print(
                f"    Early -> Late Mean CV: "
                f"{metrics['early_mean_cv']:.3f} -> "
                f"{metrics['late_mean_cv']:.3f} | "
                f"Zero-Selection Rate: "
                f"{metrics['zero_selection_rate']:.3%}"
            )


def main() -> None:
    """Run the matched Active Capacity Dense-vs-MoE Pilot."""
    args = parse_arguments()

    if args.steps <= WARMUP_STEPS:
        raise ValueError(
            f"Steps must be greater than {WARMUP_STEPS} for measurement."
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
    training_batches = make_fixed_batches(
        split="train",
        count=args.steps,
        train_data=train_data,
        validation_data=validation_data,
        batch_size=args.batch_size,
    )
    evaluation_batches = make_fixed_batches(
        split="validation",
        count=EVALUATION_BATCHES,
        train_data=train_data,
        validation_data=validation_data,
        batch_size=args.batch_size,
    )
    dense_config = build_config(
        vocab_size=len(characters),
        attention_backend=args.attention_backend,
        use_moe=False,
    )
    moe_config = build_config(
        vocab_size=len(characters),
        attention_backend=args.attention_backend,
        use_moe=True,
    )

    torch.manual_seed(deerlight.SEED)
    reference_model = deerlight.DeerlightGPTLanguageModel(
        **dense_config,
    )
    reference_state = {
        name: tensor.detach().clone()
        for name, tensor in reference_model.state_dict().items()
    }
    del reference_model

    print("Dense vs Interleaved Sparse MoE Pilot")
    print("Device:", deerlight.DEVICE)

    if deerlight.DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(deerlight.DEVICE))

    print("Steps:", args.steps)
    print("Batch Shape:", (args.batch_size, medium.MEDIUM_BLOCK_SIZE))
    print("MoE Layers:", MOE_LAYER_INDICES)
    print("Routed Experts / Top-k:", MOE_NUM_EXPERTS, "/", MOE_TOP_K)
    print("Expert Hidden Dimension:", MOE_EXPERT_HIDDEN_DIM)

    results = [
        train_configuration(
            name="Dense",
            config=dense_config,
            reference_state=reference_state,
            training_batches=training_batches,
            evaluation_batches=evaluation_batches,
        ),
        train_configuration(
            name="Interleaved MoE",
            config=moe_config,
            reference_state=reference_state,
            training_batches=training_batches,
            evaluation_batches=evaluation_batches,
        ),
    ]
    print_results(results)


if __name__ == "__main__":
    main()
