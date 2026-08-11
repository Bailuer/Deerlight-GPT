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
EMBEDDING_DIM = 64
HEAD_SIZE = 16
NUM_HEADS = 4
FEED_FORWARD_EXPANSION = 4
NUM_LAYERS = 2
GPT_LEARNING_RATE = 3e-4
GPT_TRAINING_STEPS = 3000
GPT_EVAL_INTERVAL = 300
GPT_EVAL_BATCHES = 20
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_K = 20
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "input.txt"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "deerlight_gpt_best.pt"


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

HeadKVCache = tuple[Tensor, Tensor]


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

    def forward(
        self,
        x: Tensor,
        past_key_value: HeadKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, HeadKVCache | None]:
        """Return context-aware Token Vectors and an optional updated KV Cache."""
        batch_size, query_time, _ = x.shape

        # Module 1: Project every Token Vector into Query, Key, and Value.
        # Shape of each new Tensor: (batch_size, query_time, head_size)
        queries = self.query(x)
        new_keys = self.key(x)
        new_values = self.value(x)

        # Module 2: Reuse historical Keys and Values during cached Decoding.
        past_length = 0

        if past_key_value is not None:
            past_keys, past_values = past_key_value

            if past_keys.shape != past_values.shape:
                raise ValueError("Cached Keys and Values must have the same Shape.")

            if past_keys.ndim != 3:
                raise ValueError(
                    "Cached Keys and Values must have Shape "
                    "(batch_size, past_time, head_size)."
                )

            if past_keys.shape[0] != batch_size:
                raise ValueError("KV Cache Batch Size must match the Input Batch Size.")

            if past_keys.shape[2] != self.head_size:
                raise ValueError("KV Cache Head Size must match the Attention Head Size.")

            past_length = past_keys.shape[1]
            keys = torch.cat((past_keys, new_keys), dim=1)
            values = torch.cat((past_values, new_values), dim=1)
        else:
            keys = new_keys
            values = new_values

        key_time = keys.shape[1]

        if key_time > self.causal_mask.shape[0]:
            raise ValueError(
                f"Cached Sequence Length {key_time} exceeds Block Size "
                f"{self.causal_mask.shape[0]}."
            )

        # Module 3: Compare new Queries with both historical and new Keys.
        # Shape: (batch_size, query_time, key_time)
        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores * (self.head_size ** -0.5)

        # Module 4: Select the Mask rows belonging to the new Query Positions.
        active_mask = self.causal_mask[
            past_length:past_length + query_time,
            :key_time,
        ]
        attention_scores = attention_scores.masked_fill(
            ~active_mask,
            float("-inf"),
        )

        # Module 5: Convert Scores into a Probability Distribution.
        # Every row should sum to 1.
        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        # Module 6: Compute a weighted mixture of all available Value Vectors.
        # Shape: (batch_size, query_time, head_size)
        output = attention_weights @ values

        expected_shape = (batch_size, query_time, self.head_size)
        if output.shape != expected_shape:
            raise RuntimeError(
                f"Attention Output Shape {tuple(output.shape)} does not match "
                f"the expected Shape {expected_shape}."
            )

        present_key_value = (keys, values) if use_cache else None
        return output, present_key_value


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

    original_output, _ = attention_head(sample)

    changed_sample = sample.clone()
    changed_sample[:, 2:, :] = torch.randn_like(
        changed_sample[:, 2:, :]
    )
    changed_output, _ = attention_head(changed_sample)

    assert original_output.shape == (2, 4, HEAD_SIZE)
    assert torch.isfinite(original_output).all()
    assert torch.allclose(
        original_output[:, :2, :],
        changed_output[:, :2, :],
        atol=1e-6,
    )

    # A cached one-Token Decode must match the final Position of a Full Forward.
    _, prefill_cache = attention_head(
        sample[:, :3, :],
        use_cache=True,
    )
    assert prefill_cache is not None
    assert prefill_cache[0].shape == (2, 3, HEAD_SIZE)
    assert prefill_cache[1].shape == (2, 3, HEAD_SIZE)

    cached_output, updated_cache = attention_head(
        sample[:, 3:, :],
        past_key_value=prefill_cache,
        use_cache=True,
    )
    assert updated_cache is not None
    assert updated_cache[0].shape == (2, 4, HEAD_SIZE)
    assert updated_cache[1].shape == (2, 4, HEAD_SIZE)
    assert torch.allclose(
        cached_output,
        original_output[:, 3:, :],
        atol=1e-6,
    )


# -----------------------------------------------------------------------------
# Milestone 4: Multi-Head Causal Self-Attention
# -----------------------------------------------------------------------------

MultiHeadKVCache = tuple[HeadKVCache, ...]


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

    def forward(
        self,
        x: Tensor,
        past_key_values: MultiHeadKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, MultiHeadKVCache | None]:
        """Return the combined Head Output and optional per-Head KV Caches."""
        batch_size, time, _ = x.shape

        if (
            past_key_values is not None
            and len(past_key_values) != self.num_heads
        ):
            raise ValueError(
                "Multi-Head KV Cache must contain one Cache per Attention Head."
            )

        # Module 3: Give the same Input to every Attention Head.
        head_outputs = []
        present_key_values = []

        for head_index, head in enumerate(self.heads):
            past_key_value = (
                None
                if past_key_values is None
                else past_key_values[head_index]
            )
            head_output, present_key_value = head(
                x,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            head_outputs.append(head_output)

            if use_cache:
                if present_key_value is None:
                    raise RuntimeError(
                        "Attention Head did not return a requested KV Cache."
                    )
                present_key_values.append(present_key_value)

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

        present = tuple(present_key_values) if use_cache else None
        return output, present


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

    original_output, _ = attention(sample)

    changed_sample = sample.clone()
    changed_sample[:, 2:, :] = torch.randn_like(
        changed_sample[:, 2:, :]
    )
    changed_output, _ = attention(changed_sample)

    assert len(attention.heads) == NUM_HEADS
    assert attention.head_size == EMBEDDING_DIM // NUM_HEADS
    assert original_output.shape == (2, 4, EMBEDDING_DIM)
    assert torch.isfinite(original_output).all()
    assert torch.allclose(
        original_output[:, :2, :],
        changed_output[:, :2, :],
        atol=1e-6,
    )

    # Every Head owns an independent Cache during cached Decoding.
    _, prefill_cache = attention(
        sample[:, :3, :],
        use_cache=True,
    )
    assert prefill_cache is not None
    assert len(prefill_cache) == NUM_HEADS

    for keys, values in prefill_cache:
        assert keys.shape == (2, 3, attention.head_size)
        assert values.shape == (2, 3, attention.head_size)

    cached_output, updated_cache = attention(
        sample[:, 3:, :],
        past_key_values=prefill_cache,
        use_cache=True,
    )
    assert updated_cache is not None

    for keys, values in updated_cache:
        assert keys.shape == (2, 4, attention.head_size)
        assert values.shape == (2, 4, attention.head_size)

    assert torch.allclose(
        cached_output,
        original_output[:, 3:, :],
        atol=1e-6,
    )


# -----------------------------------------------------------------------------
# Milestone 5: Transformer Block
# -----------------------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    """Process every Token independently with a nonlinear MLP.

    Input and Output Shape:
        (batch_size, time, embedding_dim)
    """

    def __init__(
        self,
        embedding_dim: int,
        expansion_factor: int,
    ) -> None:
        super().__init__()

        if expansion_factor <= 0:
            raise ValueError("Feed-Forward Expansion Factor must be positive.")

        hidden_dim = expansion_factor * embedding_dim

        self.network = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=embedding_dim,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the same MLP to every Token Vector."""
        return self.network(x)


TransformerBlockKVCache = MultiHeadKVCache


class TransformerBlock(nn.Module):
    """Combine Token communication and per-Token nonlinear processing.

    This Block uses the Pre-LayerNorm layout:
        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Input and Output Shape:
        (batch_size, time, embedding_dim)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        expansion_factor: int,
    ) -> None:
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
        )

        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
        )

    def forward(
        self,
        x: Tensor,
        past_key_values: TransformerBlockKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, TransformerBlockKVCache | None]:
        """Run both Residual Branches and return this Layer's optional Cache."""
        attention_output, present_key_values = self.attention(
            self.layer_norm_1(x),
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        x = x + attention_output
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x, present_key_values


def verify_transformer_block() -> None:
    """Check Shape, Causality, finite values, and Gradient Flow."""
    transformer_block = TransformerBlock(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        block_size=BLOCK_SIZE,
        expansion_factor=FEED_FORWARD_EXPANSION,
    )

    sample = torch.randn(
        2,
        4,
        EMBEDDING_DIM,
    )

    original_output, _ = transformer_block(sample)

    changed_sample = sample.clone()
    changed_sample[:, 2:, :] = torch.randn_like(
        changed_sample[:, 2:, :]
    )
    changed_output, _ = transformer_block(changed_sample)

    assert original_output.shape == sample.shape
    assert torch.isfinite(original_output).all()
    assert torch.allclose(
        original_output[:, :2, :],
        changed_output[:, :2, :],
        atol=1e-6,
    )

    _, prefill_cache = transformer_block(
        sample[:, :3, :],
        use_cache=True,
    )
    assert prefill_cache is not None

    cached_output, updated_cache = transformer_block(
        sample[:, 3:, :],
        past_key_values=prefill_cache,
        use_cache=True,
    )
    assert updated_cache is not None
    assert torch.allclose(
        cached_output,
        original_output[:, 3:, :],
        atol=1e-6,
    )

    test_loss = original_output.square().mean()
    test_loss.backward()

    for parameter in transformer_block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# -----------------------------------------------------------------------------
# Milestone 6: Decoder-only Deerlight GPT
# -----------------------------------------------------------------------------

ModelKVCache = tuple[TransformerBlockKVCache, ...]


class DeerlightGPTLanguageModel(nn.Module):
    """Predict the next Token with a stack of Causal Transformer Blocks."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
        expansion_factor: int,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("Number of Transformer Layers must be positive.")

        self.block_size = block_size
        self.vocab_size = vocab_size

        # Module 1: Represent both Token Identity and absolute Position.
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.position_embedding_table = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim,
        )

        # Module 2: Repeatedly refine every context-aware Token Vector.
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                block_size=block_size,
                expansion_factor=expansion_factor,
            )
            for _ in range(num_layers)
        ])

        # Module 3: Normalize the final internal Representation.
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # Module 4: Convert each Token Vector into Vocabulary Logits.
        self.language_model_head = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
        )

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
        past_key_values: ModelKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None, ModelKVCache | None]:
        """Return Logits, optional Loss, and optional per-Layer KV Caches."""
        if token_ids.ndim != 2:
            raise ValueError(
                "Token IDs must have Shape (batch_size, time)."
            )

        batch_size, time = token_ids.shape
        past_length = 0

        if past_key_values is not None:
            if len(past_key_values) != len(self.transformer_blocks):
                raise ValueError(
                    "Model KV Cache must contain one Cache per Transformer Layer."
                )

            cache_lengths = {
                keys.shape[1]
                for layer_cache in past_key_values
                for keys, _ in layer_cache
            }

            if len(cache_lengths) != 1:
                raise ValueError(
                    "Every Layer and Head KV Cache must have the same Time Length."
                )

            past_length = cache_lengths.pop()

        if past_length + time > self.block_size:
            raise ValueError(
                f"Cached Sequence Length {past_length + time} exceeds "
                f"Block Size {self.block_size}."
            )

        # Module 1: Build position-aware Token Representations.
        token_embeddings = self.token_embedding_table(token_ids)

        position_ids = torch.arange(
            past_length,
            past_length + time,
            device=token_ids.device,
        )
        position_embeddings = self.position_embedding_table(position_ids)

        x = token_embeddings + position_embeddings

        # Module 2: Let all Transformer Blocks process the Sequence.
        present_key_values = []

        for layer_index, transformer_block in enumerate(self.transformer_blocks):
            layer_past_key_values = (
                None
                if past_key_values is None
                else past_key_values[layer_index]
            )
            x, layer_present_key_values = transformer_block(
                x,
                past_key_values=layer_past_key_values,
                use_cache=use_cache,
            )

            if use_cache:
                if layer_present_key_values is None:
                    raise RuntimeError(
                        "Transformer Block did not return a requested KV Cache."
                    )
                present_key_values.append(layer_present_key_values)

        # Module 3: Produce one Vocabulary Score per Token and Position.
        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)

        expected_shape = (batch_size, time, self.vocab_size)
        if logits.shape != expected_shape:
            raise RuntimeError(
                f"GPT Logits Shape {tuple(logits.shape)} does not match "
                f"the expected Shape {expected_shape}."
            )

        loss = None

        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError(
                    "Targets must have the same Shape as Token IDs."
                )

            flat_logits = logits.reshape(
                batch_size * time,
                self.vocab_size,
            )
            flat_targets = targets.reshape(
                batch_size * time,
            )

            loss = F.cross_entropy(
                flat_logits,
                flat_targets,
            )

        present = tuple(present_key_values) if use_cache else None
        return logits, loss, present

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_kv_cache: bool = True,
    ) -> Tensor:
        """Autoregressively sample Tokens with an optional per-Layer KV Cache."""
        if temperature <= 0:
            raise ValueError("Sampling Temperature must be positive.")

        if top_k is not None and top_k <= 0:
            raise ValueError("Top-k must be positive when provided.")

        past_key_values = None

        for _ in range(max_new_tokens):
            if use_kv_cache and past_key_values is not None:
                # Decode only the newest Token; historical K/V come from Cache.
                model_input = token_ids[:, -1:]
            else:
                # Prefill the Prompt or rebuild a full sliding Context Window.
                model_input = token_ids[:, -self.block_size:]

            logits, _, past_key_values = self(
                model_input,
                past_key_values=past_key_values,
                use_cache=use_kv_cache,
            )
            final_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                active_k = min(top_k, final_logits.shape[-1])
                top_values = torch.topk(
                    final_logits,
                    k=active_k,
                    dim=-1,
                ).values
                cutoff = top_values[:, -1].unsqueeze(-1)
                final_logits = final_logits.masked_fill(
                    final_logits < cutoff,
                    float("-inf"),
                )

            probabilities = F.softmax(
                final_logits,
                dim=-1,
            )

            next_token_id = torch.multinomial(
                probabilities,
                num_samples=1,
            )
            token_ids = torch.cat(
                (token_ids, next_token_id),
                dim=1,
            )

            if use_kv_cache and past_key_values is not None:
                first_layer_cache = past_key_values[0]
                first_head_keys, _ = first_layer_cache[0]

                if first_head_keys.shape[1] >= self.block_size:
                    # Learned absolute Position Embeddings cannot append beyond
                    # block_size. Re-prefill the shifted Window next iteration.
                    past_key_values = None

        return token_ids


def verify_deerlight_gpt() -> None:
    """Check GPT Shapes, Loss, Gradient Flow, and Context Cropping."""
    vocab_size = 65

    model = DeerlightGPTLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        expansion_factor=FEED_FORWARD_EXPANSION,
    )

    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(2, 8),
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(2, 8),
    )

    logits, loss, _ = model(token_ids, targets)

    assert logits.shape == (2, 8, vocab_size)
    assert torch.isfinite(logits).all()
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Cached Model Decode must match the final Position of a Full Forward.
    _, _, prefill_cache = model(
        token_ids[:, :7],
        use_cache=True,
    )
    assert prefill_cache is not None
    assert len(prefill_cache) == NUM_LAYERS

    cached_logits, _, updated_cache = model(
        token_ids[:, 7:],
        past_key_values=prefill_cache,
        use_cache=True,
    )
    assert updated_cache is not None
    assert torch.allclose(
        cached_logits,
        logits[:, 7:],
        atol=1e-5,
    )

    # Cached and uncached greedy Generation must produce identical Tokens.
    generation_context = token_ids[:, :4]
    uncached_generated = model.generate(
        token_ids=generation_context.clone(),
        max_new_tokens=4,
        top_k=1,
        use_kv_cache=False,
    )
    cached_generated = model.generate(
        token_ids=generation_context.clone(),
        max_new_tokens=4,
        top_k=1,
        use_kv_cache=True,
    )
    assert torch.equal(cached_generated, uncached_generated)

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    long_context = torch.randint(
        low=0,
        high=vocab_size,
        size=(2, BLOCK_SIZE),
    )
    generated = model.generate(
        token_ids=long_context,
        max_new_tokens=2,
    )

    assert generated.shape == (2, BLOCK_SIZE + 2)


# -----------------------------------------------------------------------------
# Milestone 7: Deerlight GPT Training and Evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def estimate_deerlight_gpt_losses(
    model: DeerlightGPTLanguageModel,
    train_data: Tensor,
    validation_data: Tensor,
    batch_size: int,
    block_size: int,
    evaluation_batches: int,
) -> dict[str, float]:
    """Average Training and Validation Loss over multiple random Batches."""
    if evaluation_batches <= 0:
        raise ValueError("Number of Evaluation Batches must be positive.")

    was_training = model.training
    model.eval()
    model_device = next(model.parameters()).device

    estimated_losses = {}

    for split in ("train", "validation"):
        batch_losses = torch.zeros(evaluation_batches)

        for batch_index in range(evaluation_batches):
            x_batch, y_batch = get_batch(
                split=split,
                train_data=train_data,
                validation_data=validation_data,
                batch_size=batch_size,
                block_size=block_size,
            )
            x_batch = x_batch.to(model_device)
            y_batch = y_batch.to(model_device)

            _, loss, _ = model(x_batch, y_batch)

            if loss is None:
                raise RuntimeError(
                    "Evaluation Loss is None. Check the GPT Forward Pass."
                )

            batch_losses[batch_index] = loss.item()

        estimated_losses[split] = batch_losses.mean().item()

    model.train(was_training)
    return estimated_losses


def train_deerlight_gpt(
    model: DeerlightGPTLanguageModel,
    train_data: Tensor,
    validation_data: Tensor,
    learning_rate: float,
    training_steps: int,
    evaluation_interval: int,
    evaluation_batches: int,
    batch_size: int,
    block_size: int,
    checkpoint_path: Path | None = None,
    model_config: dict[str, int] | None = None,
    characters: list[str] | None = None,
) -> None:
    """Train every Deerlight GPT Parameter end to end with AdamW."""
    if training_steps <= 0:
        raise ValueError("Number of Training Steps must be positive.")

    if evaluation_interval <= 0:
        raise ValueError("Evaluation Interval must be positive.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()
    model_device = next(model.parameters()).device
    best_validation_loss = float("inf")

    if checkpoint_path is not None:
        if model_config is None or characters is None:
            raise ValueError(
                "Model Configuration and Characters are required for Checkpointing."
            )

    for step in range(1, training_steps + 1):
        x_batch, y_batch = get_batch(
            split="train",
            train_data=train_data,
            validation_data=validation_data,
            batch_size=batch_size,
            block_size=block_size,
        )
        x_batch = x_batch.to(model_device)
        y_batch = y_batch.to(model_device)

        _, training_loss, _ = model(x_batch, y_batch)

        if training_loss is None:
            raise RuntimeError(
                "Training Loss is None. Check the GPT Forward Pass."
            )

        optimizer.zero_grad(set_to_none=True)
        training_loss.backward()
        optimizer.step()

        should_evaluate = (
            step == 1
            or step % evaluation_interval == 0
            or step == training_steps
        )

        if should_evaluate:
            losses = estimate_deerlight_gpt_losses(
                model=model,
                train_data=train_data,
                validation_data=validation_data,
                batch_size=batch_size,
                block_size=block_size,
                evaluation_batches=evaluation_batches,
            )

            print(
                f"Step {step:4d}/{training_steps} | "
                f"Training Loss: {losses['train']:.4f} | "
                f"Validation Loss: {losses['validation']:.4f}"
            )

            if losses["validation"] < best_validation_loss:
                best_validation_loss = losses["validation"]

                if checkpoint_path is not None:
                    save_deerlight_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        training_loss=losses["train"],
                        validation_loss=losses["validation"],
                        model_config=model_config,
                        characters=characters,
                    )
                    print(
                        f"Saved Best Checkpoint: {checkpoint_path.name}"
                    )


# -----------------------------------------------------------------------------
# Milestone 8: Checkpointing and Sampling Controls
# -----------------------------------------------------------------------------

def save_deerlight_checkpoint(
    checkpoint_path: Path,
    model: DeerlightGPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    training_loss: float,
    validation_loss: float,
    model_config: dict[str, int],
    characters: list[str],
) -> None:
    """Atomically save Model, Optimizer, Metrics, Config, and Vocabulary."""
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "model_config": model_config,
        "characters": characters,
    }

    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_deerlight_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DeerlightGPTLanguageModel, dict]:
    """Rebuild a Deerlight GPT and restore its best saved Parameters."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model_config = checkpoint["model_config"]

    model = DeerlightGPTLanguageModel(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        block_size=model_config["block_size"],
        expansion_factor=model_config["expansion_factor"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# -----------------------------------------------------------------------------
# Program Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.set_float32_matmul_precision("high")

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

    verify_single_head_attention()
    print("\nAll Milestone 3 checks passed.")

    verify_multi_head_attention()
    print("All Milestone 4 checks passed.")

    verify_transformer_block()
    print("All Milestone 5 checks passed.")

    verify_deerlight_gpt()
    print("All Milestone 6 checks passed.")

    # Reset the Random Seed so verification does not affect Training.
    torch.manual_seed(SEED)

    model_config = {
        "vocab_size": len(characters),
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "block_size": BLOCK_SIZE,
        "expansion_factor": FEED_FORWARD_EXPANSION,
    }

    model = DeerlightGPTLanguageModel(
        vocab_size=len(characters),
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        expansion_factor=FEED_FORWARD_EXPANSION,
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    print(f"\nDeerlight GPT Parameters: {parameter_count:,}")
    print(f"Training Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    print("\nTraining Deerlight GPT...")

    train_deerlight_gpt(
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        learning_rate=GPT_LEARNING_RATE,
        training_steps=GPT_TRAINING_STEPS,
        evaluation_interval=GPT_EVAL_INTERVAL,
        evaluation_batches=GPT_EVAL_BATCHES,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        checkpoint_path=CHECKPOINT_PATH,
        model_config=model_config,
        characters=characters,
    )

    model, checkpoint = load_deerlight_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )
    print(
        f"\nLoaded Best Checkpoint from Step {checkpoint['step']} "
        f"with Validation Loss {checkpoint['validation_loss']:.4f}"
    )

    initial_token_id = stoi.get("\n", 0)
    initial_context = torch.tensor(
        [[initial_token_id]],
        dtype=torch.long,
        device=DEVICE,
    )
    generated_token_ids = model.generate(
        token_ids=initial_context,
        max_new_tokens=GENERATION_LENGTH,
        temperature=GENERATION_TEMPERATURE,
        top_k=GENERATION_TOP_K,
    )

    print(
        f"\nGenerated Text "
        f"(Temperature={GENERATION_TEMPERATURE}, "
        f"Top-k={GENERATION_TOP_K}):"
    )
    print(
        decode(
            generated_token_ids[0].detach().cpu().tolist(),
            itos,
        )
    )


if __name__ == "__main__":
    main()
