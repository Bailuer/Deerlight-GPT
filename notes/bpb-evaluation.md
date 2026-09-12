# BPB Evaluation

Run `python evaluate_bpb.py --checkpoint runs/<BPE model run>/best.pt`.
Tests: `python -m unittest test_byte_bpe test_evaluate_bpb -v`.

`BPB = total target NLL (natural logs) / target UTF-8 bytes / ln(2)`.
The Evaluator uses the checkpoint's embedded Tokenizer and verifies the corpus
hash. It scores the last 10% raw-text split, not an independent Test Set.

The first Token is context only: exclude its Bytes from the denominator. Score
every remaining Token exactly once, including the partial final block. Token
byte lengths come from Vocabulary, not individual UTF-8 decoding, since a
Token may contain only part of a Unicode Character. No Special Tokens added.

Policy: blocks of at most 64 Targets, each with 1..64 preceding Tokens. Adjacent
blocks share only the previous target as their first input. There is no KV reuse
or maximum-context sliding evaluation. This is explicit, inexpensive, and close
to training's short-block setup, but not the only legitimate evaluation policy.

For Step 3,000 Best: 57,640 Targets, 111,537 target Bytes, 210,263.577584 NLL
nats, BPB 2.719691. Uniform BPB on identical Targets is 4.651013. Mean token
NLL is 3.647876; earlier 3.6100 used only five fixed random Batches.

Do not treat byte normalization alone as a controlled Character/BPE comparison:
their first-token byte boundaries differ, and equal token context lengths cover
different raw-text ranges. Agree on a common target range and context policy
before comparing. This score uses the canonical Tokenizer encoding, not a
marginal probability summing all alternate Token sequences decoding to the text.
