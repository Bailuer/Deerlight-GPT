"""Deerlight GPT: a decoder-only Transformer built from first principles.

The first implementation stays in one file so the complete data flow remains
easy to inspect. It will be refactored only after the baseline model works.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SEED = 1337
BATCH_SIZE = 32
BLOCK_SIZE = 64
TRAIN_FRACTION = 0.9
LEARNING_RATE = 1e-2
TRAINING_STEPS = 3000
PRINT_INTERVAL = 300
GENERATION_LENGTH = 500
EMBEDDING_DIM = 32
HEAD_SIZE = 16
NUM_HEADS = 4

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
    characters = sorted(list(set(text)))

    stoi = {}
    itos = {}

    for index, character in enumerate(characters):
        stoi[character] = index
        itos[index] = character

    return characters, stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """Convert a String into a list of Token IDs."""
    token_ids = []

    for character in text:
        token_id = stoi[character]
        token_ids.append(token_id)

    return token_ids


def decode(token_ids: list[int], itos: dict[int, str]) -> str:
    """Convert a list of Token IDs back into a String."""
    characters = []

    for token_id in token_ids:
        character = itos[token_id]
        characters.append(character)

    text = "".join(characters)
    return text


def tokenize_dataset(text: str, stoi: dict[str, int]) -> Tensor:
    """Encode the complete Dataset as a one-dimensional torch.long Tensor."""
    token_ids = encode(text, stoi)

    data = torch.tensor(
        token_ids,
        dtype=torch.long,
    )

    return data


def split_dataset(
    data: Tensor,
    train_fraction: float,
) -> tuple[Tensor, Tensor]:
    """Split Token IDs into Training and Validation Sets."""
    split_index = int(
        len(data) * train_fraction
    )

    train_data = data[:split_index]
    validation_data = data[split_index:]

    return train_data, validation_data


def get_batch(
    split: str,
    train_data: Tensor,
    validation_data: Tensor,
    batch_size: int,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    """Return a random Batch of Inputs and one-token-shifted Targets."""
    # 1. Select Dataset
    if split == "train":
        source = train_data
    elif split == "validation":
        source = validation_data
    else:
        raise ValueError(
            "Split Must be 'train' or 'validation'"
        )

    # 2. Generate random starting positions
    start_indices = torch.randint(
        low=0,
        high=len(source) - block_size,
        size=(batch_size,),
    )

    x_sequences = []
    y_sequences = []

    # 3. Construct every Sequence
    for start_index in start_indices:
        start = start_index.item()

        x_sequence = source[
            start:start + block_size
        ]

        y_sequence = source[
            start + 1:start + block_size + 1
        ]

        x_sequences.append(x_sequence)
        y_sequences.append(y_sequence)

    # 4. Combine Sequences into Batches
    x = torch.stack(x_sequences)
    y = torch.stack(y_sequences)

    return x, y


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
# Milestone 2: Bigram Language Model
# -----------------------------------------------------------------------------

class BigramLanguageModel(nn.Module):
    """Predict the next Token using only the current Token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()

        # Input Token ID -> vocab_size next-Token Logits.
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=vocab_size,
        )

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return Logits and optional Cross-Entropy Loss."""
        logits = self.token_embedding_table(token_ids)

        loss = None

        if targets is not None:
            batch_size, time, vocabulary_size = logits.shape

            flat_logits = logits.reshape(
                batch_size * time,
                vocabulary_size
            )

            flat_targets = targets.reshape(
                batch_size * time
            )

            loss = F.cross_entropy(
                flat_logits,
                flat_targets,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
    ) -> Tensor:
        """Autoregressively append sampled Token IDs."""
        for _ in range(max_new_tokens):
            # Forward Pass
            logits, _ = self(token_ids)

            # Only the final Position predicts the next Token.
            final_logits = logits[:, -1, :]

            # Convert Logits to a Probability Distribution.
            probabilities = F.softmax(
                final_logits,
                dim=-1,
            )

            # Sample one Token for every Batch item.
            next_token_id = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            # Append it to the Time Dimension.
            token_ids = torch.cat(
                (token_ids, next_token_id),
                dim=1,
            )

        return token_ids


def train_bigram_model(
    model: BigramLanguageModel,
    train_data: Tensor,
    validation_data: Tensor,
) -> None:
    """Train the Bigram Model using AdamW."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    for step in range(TRAINING_STEPS):
        # Get Training Batch
        x_batch, y_batch = get_batch(
            split="train",
            train_data=train_data,
            validation_data=validation_data,
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
        )

        # Forward Pass
        _, training_loss = model(
            x_batch,
            y_batch,
        )

        if training_loss is None:
            raise ValueError(
                "Training Loss is None. Check the Model Implementation."
            )

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        training_loss.backward()

        # Parameter Update
        optimizer.step()

        # Periodic Evaluation
        if step == 0 or (step + 1) % PRINT_INTERVAL == 0:
            model.eval()

            with torch.no_grad():
                validation_x, validation_y = get_batch(
                    split="validation",
                    train_data=train_data,
                    validation_data=validation_data,
                    batch_size=BATCH_SIZE,
                    block_size=BLOCK_SIZE,
                )

                _, validation_loss = model(
                    validation_x,
                    validation_y,
                )

            if validation_loss is None:
                raise ValueError(
                    "Validation Loss is None. Check the Model Implementation."
                )

            print(
                f"Step {step + 1}/{TRAINING_STEPS} - "
                f"Training Loss: {training_loss.item():.4f} - "
                f"Validation Loss: {validation_loss.item():.4f}"
            )

            model.train()


# -----------------------------------------------------------------------------
# Milestone 3: Single-Head Causal Self-Attention
# -----------------------------------------------------------------------------

class CausalSelfAttentionHead(nn.Module):
    """Let every Token read information from itself and earlier Tokens.

    Input Shape:
        x: (batch_size, time, embedding_dim)

    Output Shape:
        output: (batch_size, time, head_size)
    """

    def __init__(
        self,
        embedding_dim: int,
        head_size: int,
        block_size: int,
    ) -> None:
        super().__init__()

        self.head_size = head_size

        # Trainable Projections: (embedding_dim -> head_size)
        self.query = nn.Linear(
            in_features=embedding_dim,
            out_features=head_size,
            bias=False,
        )
        self.key = nn.Linear(
            in_features=embedding_dim,
            out_features=head_size,
            bias=False,
        )
        self.value = nn.Linear(
            in_features=embedding_dim,
            out_features=head_size,
            bias=False,
        )

        # Fixed lower-triangular Mask. It is not a trainable Parameter.
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size,
                    dtype=torch.bool,
                )
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return context-aware Token Vectors."""
        batch_size, time, embedding_dim = x.shape

        if time > self.causal_mask.shape[0]:
            raise ValueError(
                f"Sequence Length {time} exceeds Block Size "
                f"{self.causal_mask.shape[0]}."
            )

        # Module 1: Project every Token Vector into Query, Key, and Value.
        # Shape of each: (batch_size, time, head_size)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Module 2: Compare every Query with every Key.
        # Shape: (batch_size, time, time)
        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores * (self.head_size ** -0.5)

        # Module 3: Hide future Positions from every Query.
        active_mask = self.causal_mask[:time, :time]
        attention_scores = attention_scores.masked_fill(
            ~active_mask,
            float("-inf"),
        )

        # Module 4: Convert Scores into a Probability Distribution.
        # Every row should sum to 1.
        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        # Module 5: Compute a weighted mixture of the Value Vectors.
        # Shape: (batch_size, time, head_size)
        output = attention_weights @ values

        expected_shape = (batch_size, time, self.head_size)
        if output.shape != expected_shape:
            raise RuntimeError(
                f"Attention Output Shape {tuple(output.shape)} does not match "
                f"the expected Shape {expected_shape}."
            )

        return output


def verify_single_head_attention() -> None:
    """Check Output Shape and verify that future Tokens cannot leak backward."""
    attention_head = CausalSelfAttentionHead(
        embedding_dim=EMBEDDING_DIM,
        head_size=HEAD_SIZE,
        block_size=BLOCK_SIZE,
    )

    sample = torch.randn(
        2,
        4,
        EMBEDDING_DIM,
    )

    original_output = attention_head(sample)

    changed_sample = sample.clone()
    changed_sample[:, 2:, :] = torch.randn_like(
        changed_sample[:, 2:, :]
    )
    changed_output = attention_head(changed_sample)

    assert original_output.shape == (2, 4, HEAD_SIZE)
    assert torch.isfinite(original_output).all()
    assert torch.allclose(
        original_output[:, :2, :],
        changed_output[:, :2, :],
        atol=1e-6,
    )


# -----------------------------------------------------------------------------
# Milestone 4: Multi-Head Causal Self-Attention
# -----------------------------------------------------------------------------

class MultiHeadCausalSelfAttention(nn.Module):
    """Run multiple Causal Attention Heads and combine their Outputs.

    Input Shape:
        x: (batch_size, time, embedding_dim)

    Output Shape:
        output: (batch_size, time, embedding_dim)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
    ) -> None:
        super().__init__()

        if num_heads <= 0:
            raise ValueError("Number of Attention Heads must be positive.")

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "Embedding Dimension must be divisible by Number of Heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads

        # Module 1: Create independent Attention Heads.
        self.heads = nn.ModuleList([
            CausalSelfAttentionHead(
                embedding_dim=embedding_dim,
                head_size=self.head_size,
                block_size=block_size,
            )
            for _ in range(num_heads)
        ])

        # Module 2: Mix the concatenated Head Features back into Model Space.
        self.output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the combined Output of all Attention Heads."""
        batch_size, time, _ = x.shape

        # Module 3: Give the same Input to every Attention Head.
        head_outputs = [
            head(x)
            for head in self.heads
        ]

        # Module 4: Join all Head Features along the final Dimension.
        concatenated = torch.cat(
            head_outputs,
            dim=-1,
        )

        # Module 5: Learn how to combine information from different Heads.
        output = self.output_projection(concatenated)

        expected_shape = (batch_size, time, self.embedding_dim)
        if output.shape != expected_shape:
            raise RuntimeError(
                f"Multi-Head Output Shape {tuple(output.shape)} does not match "
                f"the expected Shape {expected_shape}."
            )

        return output


def verify_multi_head_attention() -> None:
    """Check Multi-Head Output Shape and Causal Information Flow."""
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        block_size=BLOCK_SIZE,
    )

    sample = torch.randn(
        2,
        4,
        EMBEDDING_DIM,
    )

    original_output = attention(sample)

    changed_sample = sample.clone()
    changed_sample[:, 2:, :] = torch.randn_like(
        changed_sample[:, 2:, :]
    )
    changed_output = attention(changed_sample)

    assert len(attention.heads) == NUM_HEADS
    assert attention.head_size == EMBEDDING_DIM // NUM_HEADS
    assert original_output.shape == (2, 4, EMBEDDING_DIM)
    assert torch.isfinite(original_output).all()
    assert torch.allclose(
        original_output[:, :2, :],
        changed_output[:, :2, :],
        atol=1e-6,
    )


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

    model = BigramLanguageModel(vocab_size=len(characters))
    train_bigram_model(
        model=model,
        train_data=train_data,
        validation_data=validation_data,
    )

    initial_context = torch.zeros((1, 1), dtype=torch.long)
    generated_token_ids = model.generate(
        token_ids=initial_context,
        max_new_tokens=GENERATION_LENGTH,
    )

    print("\nGenerated Text:")
    print(decode(generated_token_ids[0].tolist(), itos))

    verify_single_head_attention()
    print("\nAll Milestone 3 checks passed.")

    verify_multi_head_attention()
    print("All Milestone 4 checks passed.")


if __name__ == "__main__":
    main()
