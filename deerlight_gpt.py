"""Deerlight GPT: a decoder-only Transformer built from first principles.

The first implementation stays in one file so the complete data flow remains
easy to inspect. It will be refactored only after the baseline model works.
"""

from pathlib import Path

import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SEED = 1337
BATCH_SIZE = 32
BLOCK_SIZE = 64
TRAIN_FRACTION = 0.9

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "input.txt"


# -----------------------------------------------------------------------------
# Milestone 1: Character-level Tokenizer and shifted Batch Data Pipeline
# -----------------------------------------------------------------------------

def load_text(path: Path) -> str:
    """Load the local training corpus as one String."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the training corpus at data/input.txt."
        )

    return path.read_text(encoding="utf-8")


def build_vocabulary(
    text: str,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Return Characters, string-to-integer mapping, and inverse mapping."""
    # TODO: Implement the Character Vocabulary and both mappings.
    raise NotImplementedError


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """Convert a String into a list of Token IDs."""
    # TODO: Implement Encoding.
    raise NotImplementedError


def decode(token_ids: list[int], itos: dict[int, str]) -> str:
    """Convert a list of Token IDs back into a String."""
    # TODO: Implement Decoding.
    raise NotImplementedError


def tokenize_dataset(text: str, stoi: dict[str, int]) -> Tensor:
    """Encode the complete Dataset as a one-dimensional torch.long Tensor."""
    # TODO: Encode the Dataset and convert it to the required Tensor.
    raise NotImplementedError


def split_dataset(
    data: Tensor,
    train_fraction: float,
) -> tuple[Tensor, Tensor]:
    """Split Token IDs into Training and Validation Sets."""
    # TODO: Use the first part for Training and the remainder for Validation.
    raise NotImplementedError


def get_batch(
    split: str,
    train_data: Tensor,
    validation_data: Tensor,
    batch_size: int,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    """Return a random Batch of Inputs and one-token-shifted Targets."""
    # TODO: Select the requested split.
    # TODO: Sample random valid starting positions.
    # TODO: Construct x with Shape (batch_size, block_size).
    # TODO: Construct y by shifting every Sequence forward by one Token.
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------

def verify_pipeline(
    text: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    data: Tensor,
    x_batch: Tensor,
    y_batch: Tensor,
) -> None:
    """Check the Milestone 1 invariants."""
    sample_text = text[:100]

    assert decode(encode(sample_text, stoi), itos) == sample_text
    assert data.dtype == torch.long
    assert data.ndim == 1
    assert x_batch.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert y_batch.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert torch.equal(x_batch[:, 1:], y_batch[:, :-1])


# -----------------------------------------------------------------------------
# Program Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(SEED)

    text = load_text(DATA_PATH)
    characters, stoi, itos = build_vocabulary(text)

    data = tokenize_dataset(text, stoi)
    train_data, validation_data = split_dataset(data, TRAIN_FRACTION)

    x_batch, y_batch = get_batch(
        split="train",
        train_data=train_data,
        validation_data=validation_data,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
    )

    verify_pipeline(
        text=text,
        stoi=stoi,
        itos=itos,
        data=data,
        x_batch=x_batch,
        y_batch=y_batch,
    )

    print("Deerlight GPT - Milestone 1")
    print("Dataset Length:", len(text))
    print("Vocabulary Size:", len(characters))
    print("Training Tokens:", len(train_data))
    print("Validation Tokens:", len(validation_data))
    print("Input Batch Shape:", tuple(x_batch.shape))
    print("Target Batch Shape:", tuple(y_batch.shape))
    print("\nFirst Input Sequence:")
    print(decode(x_batch[0].tolist(), itos))
    print("\nFirst Target Sequence:")
    print(decode(y_batch[0].tolist(), itos))
    print("\nAll Milestone 1 checks passed.")


if __name__ == "__main__":
    main()
