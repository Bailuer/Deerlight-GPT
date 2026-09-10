# BPE Model pilot

Run `python train_bpe.py --tokenizer-dir runs/<BPE experiment directory> --steps 30`
after `python experiments/evaluate_bpe.py`.

The independent entry point preserves the original character model. Its small
FP32 configuration is a functional pilot, not a Medium quality benchmark.

- Verify corpus hash and the original raw-text split before encoding.
- Cache both splits using a corpus/Tokenizer/split fingerprint.
- Build `torch.long` shifted batches: `x[:, 1:] == y[:, :-1]`.
- Use actual Tokenizer vocabulary size for Embedding and Vocabulary Projection.
- Train with AdamW and gradient clipping; keep evaluation RNG separate.
- Evaluate fixed Training and Validation batches every `--eval-interval` steps
  (default 250), with five batches per split and separate RNG streams.
- Save `best.pt` selected by Validation Loss and `final.pt` in a unique `runs`
  directory. `history.json` records evaluated Losses; `sample.txt` uses best.
- Embed Tokenizer JSON in the checkpoint alongside Model config and Weights.
- Reload final and check identical Logits and Token IDs; reload best and verify
  its Validation Loss, then generate 100 Tokens with sampling seed 1341.

No BOS/EOS/PAD IDs are inserted in this continuous-text pilot. They remain in
the Vocabulary but their intended behavior has not been trained. Generation
is fixed-length, not EOS-terminated. Invalid generated UTF-8 is replaced only
for display. A 30-step sample is not expected to be coherent.

Raw BPE per-token Loss cannot be directly compared with character-level Loss.
Use a held-out byte-normalized evaluation for a later quality comparison.
