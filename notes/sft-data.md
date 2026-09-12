# SFT Data Collator

Run `python sft_data.py`; tests: `python -m unittest test_sft_data -v`.

`SFTDataCollator(tokenizer)(examples)` returns `input_ids` and `labels`.
Each example is a dictionary with string `user` and `assistant` fields.

1. Prompt: BOS ID + ordinary text `<USER>\n{user}\n<EOT>\n<ASSISTANT>\n`.
2. Answer: separately encoded answer + EOS ID.
3. Concatenate; mask Prompt labels, then shift Inputs/Labels exactly once.
4. Right-pad Inputs with PAD ID, Labels with -100, and Stack.
5. Call our model as `model(batch['input_ids'], batch['labels'])`, not `model(**batch)`.

Role strings are NOT new atomic Special Tokens. Literal user text can contain
these strings; this teaching format is not an injection-resistant boundary.
Prompt and answer are encoded separately to preserve the supervision boundary;
this may differ from encoding the whole concatenated string. Generation must
use `encode_prompt` for the same prefix convention. Existing Vocabulary/Weights
are not resized, and this module does not train a Model.

Only Single-turn, Right Padding is supported. Samples longer than max_length
raise an error so EOS/answer supervision is not silently lost. An empty answer
still supervises EOS. No Packing or left-padding mask support is implied.
Real positions cannot attend future PADs in this causal setup; PAD queries are
ignored by Loss. This does not justify omitting padding masks in other setups.

In the Byte-only demo, Hello contributes 5 byte Targets + EOS = 6; Hello friend
contributes 12 + EOS = 13. Global valid-token mean uses 19, not Sample count 2.
With a trained BPE, counts depend on its Rules. The earlier 2+3 example assumed
whole-word Tokens and was not the actual Tokenizer output.
