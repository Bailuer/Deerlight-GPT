"""Benchmark manual Attention and available PyTorch SDPA CUDA backends."""

from contextlib import nullcontext
from pathlib import Path
import sys
import time
import warnings

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import deerlight_gpt as deerlight  # noqa: E402


BATCH_SIZE = 32
SEQUENCE_LENGTH = 256
EMBEDDING_DIM = 384
NUM_QUERY_HEADS = 6
NUM_KV_HEADS = 2
WARMUP_STEPS = 10
MEASURED_STEPS = 50


def synchronize() -> None:
    """Wait for queued CUDA work before reading the Wall-clock Timer."""
    torch.cuda.synchronize()


def backend_context(backend: SDPBackend | None):
    """Select one SDPA CUDA implementation or leave manual code unchanged."""
    if backend is None:
        return nullcontext()

    return sdpa_kernel(backend)


def run_step(
    attention: deerlight.MultiHeadCausalSelfAttention,
    sample: torch.Tensor,
    backend: SDPBackend | None,
) -> None:
    """Run one BF16 Forward and Backward without an Optimizer update."""
    attention.zero_grad(set_to_none=True)

    if sample.grad is not None:
        sample.grad = None

    with backend_context(backend):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, _ = attention(sample)
            loss = output.square().mean()

        loss.backward()


def measure_backend(
    name: str,
    attention_backend: str,
    forced_backend: SDPBackend | None,
) -> dict[str, float | str]:
    """Measure Training latency and allocated VRAM for one implementation."""
    torch.manual_seed(deerlight.SEED)
    torch.cuda.manual_seed_all(deerlight.SEED)

    attention = deerlight.MultiHeadCausalSelfAttention(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_QUERY_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        block_size=SEQUENCE_LENGTH,
        attention_backend=attention_backend,
    ).cuda()
    sample = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
        device="cuda",
        requires_grad=True,
    )

    for _ in range(WARMUP_STEPS):
        run_step(attention, sample, forced_backend)

    synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for _ in range(MEASURED_STEPS):
        run_step(attention, sample, forced_backend)

    synchronize()
    elapsed = time.perf_counter() - start
    peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3

    del sample
    del attention
    torch.cuda.empty_cache()

    return {
        "name": name,
        "mean_step_ms": elapsed / MEASURED_STEPS * 1_000,
        "peak_allocated_gib": peak_allocated_gib,
    }


def flash_attention_available() -> bool:
    """Return whether this PyTorch Build can execute Flash Attention."""
    attention = deerlight.MultiHeadCausalSelfAttention(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_QUERY_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        block_size=SEQUENCE_LENGTH,
        attention_backend="sdpa",
    ).cuda()
    sample = torch.randn(
        1,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
        device="cuda",
        requires_grad=True,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_step(attention, sample, SDPBackend.FLASH_ATTENTION)
        synchronize()
        return True
    except RuntimeError:
        return False
    finally:
        del sample
        del attention
        torch.cuda.empty_cache()


def main() -> None:
    """Run the controlled CUDA Backend comparison."""
    if not torch.cuda.is_available():
        raise RuntimeError("This Benchmark requires a CUDA GPU.")

    configurations = (
        ("Manual", "manual", None),
        ("SDPA Math", "sdpa", SDPBackend.MATH),
        ("SDPA cuDNN", "sdpa", SDPBackend.CUDNN_ATTENTION),
    )
    results = [
        measure_backend(name, attention_backend, forced_backend)
        for name, attention_backend, forced_backend in configurations
    ]

    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Flash Attention executable:", flash_attention_available())
    print("Shape (B, T, C):", (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM))
    print("Query Heads / KV Heads:", NUM_QUERY_HEADS, "/", NUM_KV_HEADS)
    print("| Backend | Mean Forward+Backward (ms) | Peak Allocated VRAM (GiB) |")
    print("|---|---:|---:|")

    for result in results:
        print(
            f"| {result['name']} | "
            f"{result['mean_step_ms']:.3f} | "
            f"{result['peak_allocated_gib']:.3f} |"
        )


if __name__ == "__main__":
    main()
