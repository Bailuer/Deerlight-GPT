"""Single-turn teaching Collator: right padding, explicit shift, assistant loss.

Role markers are ordinary text, NOT new Special Token IDs. Encode the prompt
and answer separately so a BPE merge cannot cross the supervision boundary.
Generation must use the same encode_prompt function. No packing or truncation.
"""

import torch
from byte_bpe import ByteBPETokenizer

IGNORE_INDEX = -100


def encode_prompt(tokenizer: ByteBPETokenizer, user: str) -> list[int]:
    if not isinstance(user, str):
        raise TypeError("user must be a string")
    # These readable delimiters are a teaching format, not a security boundary.
    return [tokenizer.BOS_ID] + tokenizer.encode(
        f"<USER>\n{user}\n<EOT>\n<ASSISTANT>\n"
    )


def encode_example(tokenizer, user, assistant, max_length):
    if not isinstance(assistant, str):
        raise TypeError("assistant must be a string")
    prompt = encode_prompt(tokenizer, user)
    answer = tokenizer.encode(assistant) + [tokenizer.EOS_ID]
    sequence = prompt + answer
    # Labels first align with their own tokens, then shift with the sequence.
    unshifted_labels = [IGNORE_INDEX] * len(prompt) + answer
    input_ids = sequence[:-1]
    labels = unshifted_labels[1:]
    if len(input_ids) > max_length:
        raise ValueError("Sample exceeds max_length; shorten it explicitly (no silent truncation)")
    return input_ids, labels


class SFTDataCollator:
    def __init__(self, tokenizer, max_length=64):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        if not examples:
            raise ValueError("Batch must not be empty")
        rows = [encode_example(self.tokenizer, item["user"], item["assistant"],
                               self.max_length) for item in examples]
        width = max(len(x) for x, _ in rows)
        inputs = torch.full((len(rows), width), self.tokenizer.PAD_ID, dtype=torch.long)
        labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
        for row, (x, y) in enumerate(rows):
            inputs[row, :len(x)] = torch.tensor(x, dtype=torch.long)
            labels[row, :len(y)] = torch.tensor(y, dtype=torch.long)
        # For this right-padded causal model, real tokens cannot see later PADs.
        # Do not reuse this assumption for left padding or packed samples.
        return {"input_ids": inputs, "labels": labels}


if __name__ == "__main__":
    tokenizer = ByteBPETokenizer(259)  # Byte-only demo makes counts transparent.
    batch = SFTDataCollator(tokenizer)([
        {"user": "Hi", "assistant": "Hello"},
        {"user": "Hi", "assistant": "Hello friend"},
    ])
    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("labels shape:", tuple(batch["labels"].shape))
    for i, labels in enumerate(batch["labels"]):
        valid = labels[labels != IGNORE_INDEX].tolist()
        print(f"Sample {i}: target IDs={valid}, count={len(valid)}")
        print("Supervised text:", tokenizer.decode(valid))
    print("Batch valid Targets:", (batch["labels"] != IGNORE_INDEX).sum().item())
