---
title: "Drop-Upcycling and the Birth of Zen MoDE Architecture"
date: "2026-02-28"
description: "How Drop-Upcycling (arXiv:2502.19261) transforms dense checkpoints into MoE models at 1/4 training cost, and how it shapes Zen MoDE — our Mixture of Distilled Experts architecture."
tags: ["Research", "MoE", "Architecture", "Drop-Upcycling", "Zen MoDE"]
---


<LinkButton href="https://arxiv.org/abs/2502.19261" label="DROP-UPCYCLING PAPER" external />
<LinkButton href="https://huggingface.co/zenlm" label="ZEN MODELS" external />
<LinkButton href="https://github.com/zenlm" label="ZEN CODE" external />

Mixture of Experts (MoE) is the architecture that makes trillion-parameter models economically viable. By routing each token through a small subset of expert networks rather than the full parameter set, MoE achieves large-model quality at dense-model inference cost. The problem: training an MoE from scratch is expensive. You are paying for both the scale and the specialization overhead.

Drop-Upcycling is a technique that converts a trained dense checkpoint into an MoE at roughly 1/4 the training cost of building the MoE from scratch. It is one of the foundational techniques behind **Zen MoDE** — our Mixture of Distilled Experts architecture. This post explains how it works, why it works, and how we apply it at three scales.

## Why Naive Expert Cloning Fails

The obvious approach to dense-to-MoE conversion: clone the dense FFN block N times to create N experts, initialize a router, and continue training. This costs almost nothing upfront. The problem reveals itself within a few thousand steps: **weight correlation collapse**.

When all experts start from identical weights, they receive identical gradients on every token that routes to multiple of them simultaneously. The router has no signal to differentiate them. Gradient updates push all experts in the same direction. Within tens of thousands of steps, the experts have converged to nearly identical weights despite being nominally separate. The MoE behaves like a dense model with routing overhead and no specialization benefit.

This is not a training instability — it is a symmetry problem. Identical initialization creates a saddle point in the loss landscape where all expert-breaking perturbations are equally likely but none are preferred. The model sits at the saddle indefinitely.

## Drop-Upcycling: Breaking Symmetry With Structured Noise

**Paper**: arXiv:2502.19261

Drop-Upcycling solves the symmetry problem by deliberately damaging each expert's initialization in a structured way. For each expert i, randomly select p% of the FFN rows and reinitialize them from a normal distribution:

```
w_j^(i) = N(0, σ²)   if j ∈ dropped_rows(i)
w_j^(i) = w_j^dense  otherwise
```

The dropped set is different for each expert (sampled independently). This breaks the symmetry: experts start from the same functional foundation but with different "holes" in their weight matrices.

```python
import torch
import torch.nn as nn
from typing import List

def drop_upcycle(
    dense_ffn: nn.Linear,
    n_experts: int,
    drop_rate: float = 0.1,
    init_std: float = 0.02,
) -> List[nn.Linear]:
    """
    Convert a single dense FFN layer into n_experts expert layers
    using Drop-Upcycling initialization.

    Args:
        dense_ffn: Pretrained dense FFN layer
        n_experts: Number of MoE experts to create
        drop_rate: Fraction of rows to reinitialize per expert
        init_std: Standard deviation for reinitialized rows

    Returns:
        List of n_experts initialized expert layers
    """
    d_out, d_in = dense_ffn.weight.shape
    n_drop = max(1, int(d_out * drop_rate))
    experts = []

    for i in range(n_experts):
        expert = nn.Linear(d_in, d_out, bias=dense_ffn.bias is not None)
        with torch.no_grad():
            # Start from dense weights
            expert.weight.copy_(dense_ffn.weight)
            if dense_ffn.bias is not None:
                expert.bias.copy_(dense_ffn.bias)

            # Drop a unique random subset of rows
            drop_indices = torch.randperm(d_out)[:n_drop]
            expert.weight[drop_indices] = torch.randn(n_drop, d_in) * init_std
            if dense_ffn.bias is not None:
                expert.bias[drop_indices] = 0.0

        experts.append(expert)

    return experts


def upcycle_transformer_block(
    dense_block,
    n_experts: int,
    drop_rate: float = 0.1,
) -> dict:
    """Upcycle a full transformer FFN block into MoE experts."""
    return {
        'gate_proj': drop_upcycle(dense_block.gate_proj, n_experts, drop_rate),
        'up_proj':   drop_upcycle(dense_block.up_proj,   n_experts, drop_rate),
        'down_proj': drop_upcycle(dense_block.down_proj, n_experts, drop_rate),
    }
```

The `drop_rate` hyperparameter is the key dial. Too low (< 5%) and experts remain too correlated. Too high (> 30%) and you lose the functional initialization benefit — the expert essentially starts from random weights. The sweet spot we found empirically: **8-12% for general language models, 15% for code-specialized models** (where more aggressive diversity is needed to separate syntax vs. semantics experts).

## Training Dynamics: Implicit Specialization Signal

Why do dropped experts specialize rather than just learn to patch their own holes? The mechanism is elegant.

After the first few training steps, the "intact" experts (those with more of the original dense weights) perform better on common tokens — they have a head start. The router, which is optimizing for overall performance, learns to send common tokens to the better-performing intact experts. The dropped experts receive a different token distribution: harder tokens, rarer constructs, edge cases that the intact experts handle poorly.

This is the **implicit specialization signal**: experts do not specialize by design, they specialize by default. Each expert optimizes for the token distribution it actually receives, and that distribution is different for each expert because their relative competencies differ. By 50K training steps, the expert specialization is measurable:

- Intact experts (low drop rate) converge toward high-frequency, syntactic functions
- Heavily-dropped experts develop novel representations for rare or complex tokens

The dense-to-MoE transition effectively turns a capability gap (some experts start worse) into a specialization signal (those experts become domain-specific).

## Results at Scale

On the primary benchmark suite (comparing Drop-Upcycled MoE vs. MoE trained from scratch):

- **5.9B active parameters**: Drop-Upcycled MoE achieves 13B-equivalent quality at 1/4 the training FLOPs
- **MMLU**: Drop-Upcycling reaches 75.4 vs. from-scratch MoE at 74.8 (at same FLOP budget)
- **HumanEval**: 68.2 vs. 65.1 — Drop-Upcycling is better here because code has cleaner specialization axes
- **Training efficiency**: 4x speedup to target quality vs. from-scratch MoE

The 1/4 FLOP claim requires context: the dense checkpoint training cost is amortized. If you already have the dense model (which you do, because you trained it first), the incremental cost to get an MoE is roughly 1/4 of a from-scratch MoE run. The total cost (dense + MoE) is higher than from-scratch, but for organizations that already have dense checkpoints — which is everyone — the marginal cost argument is what matters.

## Zen MoDE: Three Scales of Application

Zen MoDE (Mixture of Distilled Experts) applies Drop-Upcycling at three scales:

**zen4-mini (4B total, 4B active)** — Dense. No upcycling needed at this scale; the routing overhead would dominate the compute savings. Zen4-mini uses a dense Qwen3 base.

**zen4-max (30B total, 3B active)** — 16 experts, 2 active per token. Drop-Upcycled from an 8B dense checkpoint. Drop rate: 8%. Router: learned top-2 routing with load balancing. The transition from 8B dense to 30B MoE takes 200M training tokens, roughly 3 days on 32×H100.

**zen4-ultra (1T total, 32B active)** — 384 experts, 8 active per token. This is our frontier model based on the Kimi K2.5 architecture. The upcycling here was done by the upstream team; we train behavioral adapters on top using GT-QLoRA (see the companion post on that technique).

## What Do the Experts Actually Learn?

We ran expert attribution analysis on zen4-max after 500M post-upcycling training tokens. The methodology: for each expert, collect the 10K tokens that activate it most strongly and analyze the distribution.

The results cluster into recognizable domains:

| Expert Group | Token Characteristics |
|---|---|
| Experts 0-3 | High-frequency English function words, punctuation |
| Experts 4-6 | Code tokens: brackets, operators, keywords |
| Experts 7-9 | Mathematical notation, numerals, equations |
| Experts 10-12 | Multilingual tokens (Chinese, Arabic, Cyrillic) |
| Experts 13-15 | Rare English words, technical terminology |

This is not designed specialization — it emerged from the implicit signal described above. The router discovered that routing code tokens to experts 4-6 produces better outputs than routing them to experts 0-3. No explicit supervision was provided.

## Q-GaLore for MoE Training Efficiency

Training a 30B MoE requires careful memory management. We use Q-GaLore (Quantized Gradient Low-Rank Projection) for the upcycling phase:

- Gradient projection: instead of storing full gradients for all parameters, project them into a low-rank subspace (rank 128 for most layers)
- Quantize the projected gradients to INT8 before accumulation
- Result: 50% memory reduction vs. standard LoRA, with +5.19 MMLU points vs. QLoRA on equivalent compute

The memory savings matter because upcycling requires loading both the dense checkpoint (for initialization) and the growing MoE checkpoint (for training) simultaneously. Q-GaLore makes this tractable on 8×A100 80GB configurations where it would otherwise OOM.

## Research Frontier: Progressive Router Pruning

The current Drop-Upcycling approach creates experts of fixed capacity. An open question we are actively investigating: can you progressively prune the router to identify which experts are actually being used, then collapse unused capacity back into the shared parameters?

Early results suggest that after 500M training tokens, roughly 20-30% of experts receive less than 2% of routing probability across the evaluation corpus. These "dormant" experts can be pruned and their parameters absorbed back into the shared FFN without measurable quality degradation. This gives a dynamic MoE that starts dense, develops expert structure, and then self-compresses to its natural capacity.

The mechanism matters for continual learning: as new domains are added, dormant experts can be "awakened" and repurposed for the new domain rather than creating new experts from scratch. This connects Drop-Upcycling directly to the SuRe + OPCM continual learning stack described in our companion post.

---

*Zen LM is a joint initiative of Hanzo AI Inc. (Techstars '17) and Zoo Labs Foundation (501c3).*
