# Mini IsoFLOP Scaling Pilot

## Question

At a fixed approximate Dense Training budget `N * D = 2e11`, how should
Deerlight GPT allocate resources between Model Parameters and Character Token
presentations?

## Controlled setup

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- Seed: 1337
- Approximate Training Compute: `6 * N * D = 1.2e12 FLOPs`
- Batch Shape: `(32,64)`
- Tokenizer: Character-level
- Underlying Training Corpus: 1,003,854 Character Tokens
- Precision: BF16 Autocast
- Optimizer: AdamW
- Learning Rate: 0.0003 for all sizes
- Gradient Clipping Max Norm: 1.0
- Validation: the same 20 fixed Batches for all Models, generated from a
  Validation-specific Seed independent of the Training stream

Only Model width, depth, and the inversely computed Training Token budget vary.

## Results

| Model | C | Layers | Parameters | Steps | Training Tokens | Approx FLOPs | Tokens/s | Wall s | Peak GiB | Final Train Loss | Final Val Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 32 | 1 | 16,641 | 5,868 | 12,017,664 | 1.200e12 | 417,751 | 29.97 | 0.068 | 2.0602 | 2.1375 |
| Mini | 48 | 2 | 55,793 | 1,750 | 3,584,000 | 1.200e12 | 272,119 | 13.55 | 0.076 | 1.9685 | 2.0151 |
| Small | 64 | 2 | 100,993 | 967 | 1,980,416 | 1.200e12 | 272,105 | 7.54 | 0.081 | 2.0460 | 2.0629 |
| Medium | 96 | 3 | 308,417 | 317 | 649,216 | 1.201e12 | 192,970 | 3.49 | 0.101 | 2.1447 | 2.2086 |
| Large | 128 | 4 | 722,881 | 135 | 276,480 | 1.199e12 | 153,088 | 1.90 | 0.134 | 2.3728 | 2.4283 |

## Interpretation

The initial three-size Profile placed its best point at the smallest tested
Model, so the search interval was extended below 100K Parameters. The expanded
five-size Profile formed the expected U-shaped curve. Mini achieved the lowest
Validation Loss at 55,793 Parameters. Tiny became capacity-limited despite
receiving 12.0M Token presentations, while Models above 100K became increasingly
undertrained at the fixed Compute budget.

The experiment therefore brackets a local Compute-Optimal region between the
16,641-Parameter Tiny Model and the 100,993-Parameter Small Model. Mini is the
best tested discrete configuration, not a claim that 55,793 is the exact or
globally optimal Parameter count.

The result must not be generalized beyond this Pilot. All configurations used
one Learning Rate, one Seed, one Character Dataset, and an approximate Compute
formula based on Total Parameters. Tiny consumed almost twelve times as many
Token presentations as the underlying corpus contains, while Large consumed
only about 27.5% of one corpus-length.

Despite nearly identical estimated FLOPs, measured Wall-clock time ranged from
1.90 to 29.97 seconds. Larger matrix multiplications utilized the GPU more
efficiently, while Tiny required many more Steps and paid more Python,
Kernel-launch, Optimizer, and data-pipeline overhead. Equal theoretical FLOPs do
not imply equal Hardware time.

## Next experiment

Repeat the five-size Profile across multiple Compute budgets. Each budget yields
one locally optimal Model/Data allocation; those optima can then be fitted to
estimate how optimal Parameters and Tokens change with Compute. A stronger study
must also tune Learning Rate by scale and use multiple Random Seeds.

## Reproduce

```powershell
python experiments/isoflop_scaling_pilot.py
```
