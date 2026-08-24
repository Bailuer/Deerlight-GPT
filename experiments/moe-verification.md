# Sparse MoE Architecture Verification

## Question

Can Deerlight GPT replace selected dense SwiGLU Feed-Forward branches with a
transparent Sparse Mixture of Experts layer while preserving model Shapes and
Gradient Flow?

## Architecture

- A learned Linear Router produces one score per Routed Expert.
- Loss-free Routing Bias affects Top-k selection but not combination weights.
- Each selected Routed Expert receives the complete Token Representation.
- Selected Expert outputs are weighted and combined with `index_add_`.
- One Shared Expert processes every Token without Router selection.
- Per-Expert assignment counts are retained as non-persistent monitoring state.

## Required invariants

- Input and output Shapes are both `(B,T,C)`.
- Total Routed assignments equal `B * T * Top-k`.
- Router, selected Routed Experts, Shared Expert, and Input receive finite
  Gradients.
- Routing Bias is checkpointed but is not an AdamW-trained Parameter.
- Evaluation records Expert Loads without updating Routing Bias.
- Dense and MoE Blocks can coexist in one Transformer stack.

## Limitations

This implementation intentionally uses a Python Expert loop and dynamic Gather
operations for clarity. It does not yet implement fixed Expert Capacity,
Token Dropping, grouped GEMMs, Expert Parallelism, or All-to-All communication.
Those are systems optimizations rather than changes to the MoE mathematics.
