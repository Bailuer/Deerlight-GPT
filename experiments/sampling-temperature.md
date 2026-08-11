# Sampling Temperature with Fixed Top-k

Date: 2026-08-10

## Question

How does Sampling Temperature change generation quality when the Model,
Checkpoint, Prompt, Random Seed, and Top-k remain fixed?

## Controlled Variables

- Checkpoint Step: 3,000
- Validation Loss: 2.1022
- Embedding Dimension: 64
- Parameters: 112,193
- Initial Token: newline
- Random Seed: 1337, reset before every Sample
- Generated Tokens per Sample: 300
- Top-k: 20

## Changed Variable

```text
Temperature: 0.6, 0.8, 1.0, 1.2
```

## Observations

| Temperature | Observation |
|---:|---|
| 0.6 | More repetitive, safer common Character patterns, less variety |
| 0.8 | Best balance of recognizable words, structure, and variation |
| 1.0 | More diverse but less stable spelling and sentence structure |
| 1.2 | Most diverse, with more malformed names and noisy transitions |

Representative beginnings:

```text
0.6: CANENS: Sterecow and is so be madise...
0.8: CANED If this own, havere, beve a sell of oure...
1.0: CANED If this own, havers, beve a selay...
1.2: CANCIORIRD: Olow aff is so beve a selay!...
```

## Result

Temperature 0.8 with Top-k 20 is the default for this Checkpoint. Lower
Temperature made the Model repetitive, while higher Temperature exposed more
of its uncertain low-quality predictions.

## Interpretation

Temperature changes the Sampling Distribution rather than Model knowledge.
Improved-looking Output at a lower Temperature does not imply a lower
Validation Loss or a more capable Model.

## Limitations

- Only one Checkpoint and Random Seed were tested.
- Quality was judged qualitatively.
- Character-level Tokenization makes spelling stability highly sensitive to
  individual Sampling decisions.
