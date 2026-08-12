"""Compare MHA, GQA, and MQA Parameters, KV Cache bytes, and latency."""

from pathlib import Path
import sys
import time

import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import deerlight_gpt as deerlight  # noqa: E402


WARMUP_RUNS = 2
MEASURED_RUNS = 10
GENERATED_TOKENS = 60


def synchronize() -> None:
    """Wait for asynchronous CUDA work before reading the Wall-clock Timer."""
    if deerlight.DEVICE.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_configuration(
    name: str,
    num_kv_heads: int,
) -> dict[str, float | int | str]:
    """Measure one randomly initialized Attention configuration."""
    torch.manual_seed(deerlight.SEED)

    model = deerlight.DeerlightGPTLanguageModel(
        vocab_size=65,
        embedding_dim=deerlight.EMBEDDING_DIM,
        num_heads=deerlight.NUM_HEADS,
        num_kv_heads=num_kv_heads,
        num_layers=deerlight.NUM_LAYERS,
        block_size=deerlight.BLOCK_SIZE,
        feed_forward_dim=deerlight.calculate_swiglu_hidden_dim(
            deerlight.EMBEDDING_DIM,
        ),
    ).to(deerlight.DEVICE)
    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    full_context = torch.zeros(
        (1, deerlight.BLOCK_SIZE),
        dtype=torch.long,
        device=deerlight.DEVICE,
    )
    _, _, past_key_values = model(
        full_context,
        use_cache=True,
    )

    if past_key_values is None:
        raise RuntimeError("Model did not return a requested KV Cache.")

    cache_bytes = sum(
        keys.numel() * keys.element_size()
        + values.numel() * values.element_size()
        for keys, values in past_key_values
    )

    prompt = torch.zeros(
        (1, 1),
        dtype=torch.long,
        device=deerlight.DEVICE,
    )

    for _ in range(WARMUP_RUNS):
        model.generate(
            prompt.clone(),
            max_new_tokens=GENERATED_TOKENS,
            top_k=1,
            use_kv_cache=True,
        )
    synchronize()

    start_time = time.perf_counter()

    for _ in range(MEASURED_RUNS):
        model.generate(
            prompt.clone(),
            max_new_tokens=GENERATED_TOKENS,
            top_k=1,
            use_kv_cache=True,
        )

    synchronize()
    mean_latency = (
        time.perf_counter() - start_time
    ) / MEASURED_RUNS

    return {
        "name": name,
        "num_kv_heads": num_kv_heads,
        "parameters": parameter_count,
        "cache_bytes": cache_bytes,
        "mean_latency_ms": mean_latency * 1_000,
    }


def main() -> None:
    """Run and print the controlled Architecture comparison."""
    configurations = (
        ("MHA", deerlight.NUM_HEADS),
        ("GQA", deerlight.NUM_HEADS // 2),
        ("MQA", 1),
    )
    results = [
        measure_configuration(name, num_kv_heads)
        for name, num_kv_heads in configurations
    ]

    print(f"Device: {deerlight.DEVICE}")
    print(
        "| Architecture | KV Heads | Parameters | "
        "KV Cache Bytes | Mean Latency (ms) |"
    )
    print("|---|---:|---:|---:|---:|")

    for result in results:
        print(
            f"| {result['name']} | {result['num_kv_heads']} | "
            f"{result['parameters']:,} | {result['cache_bytes']:,} | "
            f"{result['mean_latency_ms']:.3f} |"
        )


if __name__ == "__main__":
    main()
