# Color translation SFT pilot

Date: 2026-09-13. Script: `train_sft_colors.py`.

Run from the project root:

```powershell
python train_sft_colors.py --checkpoint runs/bpe-model-20260911-035646/best.pt --steps 800
```

## Setup

- Author-created teaching pairs; no downloaded instruction corpus.
- Four seen colors: red/blue/green/black -> 红色/蓝色/绿色/黑色.
- Training: `Translate {c} into Chinese.` and `Chinese for {c}?` (8 Samples).
- Validation: `What is {c} in Chinese?` (4 Samples). Template held out,
  not color or answer held out. This Validation set is used for selection,
  so it is not an independent Test Set.
- Initialize from the 3,000-step BPE Pretraining checkpoint. Same 512-token
  Vocabulary and architecture, full Parameter updates; new AdamW state.
- FP32 CUDA, seed 1342, lr=3e-4, gradient clipping 1.0; full Batch of 8
  repeated for 800 Steps. Assistant-only Labels, EOS included, PAD ignored.
- Ordinary text role markers; unchanged IDs. No Model capability guarantee
  follows from adding these strings; SFT must learn their use.
- Every 200 Steps evaluate complete splits under Teacher Forcing and run
  greedy Generation from prompt only, stopping at EOS or 16 new Tokens.
- Exact Match uses decoded answer verbatim, no stripping/case normalization.
  EOS rate is measured separately, with no answer or EOS forced at decoding.
- Best is selected by Validation token Loss; selected checkpoint is reloaded
  and its Loss and Validation outputs verified.

## Results

| Step | Train Loss | Val Loss | Train EM | Val EM | Val EOS |
|---|---:|---:|---:|---:|---:|
| 0 | 12.1341 | 12.1610 | 0% | 0% | 0% |
| 200 | 0.0223 | 0.0747 | 100% | 75% | 100% |
| 400 | 0.0076 | 0.0746 | 100% | 75% | 100% |
| 600 | 0.0041 | 0.0835 | 100% | 75% | 100% |
| 800 | 0.0026 | 0.0919 | 100% | 75% | 100% |

Best Step 400 answers the unseen template for blue, green and black correctly,
but answers `What is red in Chinese?` with `绿色` instead of `红色`.
All eight Training prompts are correct at this step and terminate with EOS.

This verifies fitting the task and learned termination, with limited evidence
of transfer to one new wording. It does not establish general translation,
conversational competence, or robust generalization. Only one Seed and four
Validation Samples were tested. Later Train/Val divergence suggests overfitting
under this metric; repeated training did not fix the held-out red error.
No old language-capability retention evaluation was performed.

Outputs: `runs/sft-colors-20260913-045832/` with dataset, full history,
best checkpoint and best metrics; ignored by Git. The source Pretraining
checkpoint is unchanged. The usual 12 BPE/BPB/Collator tests passed.
