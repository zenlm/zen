---
title: "7680-Dimensional Embeddings: More Dimensions, Better Retrieval"
date: "2022-12-05"
description: "Why we trained embedding models with 7680 dimensions and what we learned about the relationship between dimensionality and retrieval quality."
author: "Zach Kelling"
tags: ["Research", "Embeddings", "Retrieval"]
---


Embedding dimensions have standardized around powers of two: 768, 1536, occasionally 4096. We asked a simple question: what happens if we go bigger? The answer surprised us.

## Background: Why Dimensions Matter

Text embeddings map variable-length sequences to fixed-dimensional vectors. These vectors enable semantic similarity search, clustering, and retrieval. The dimension count determines the vector space's capacity.

Lower dimensions mean:
- Smaller storage requirements
- Faster similarity computations
- Potential information loss

Higher dimensions mean:
- More expressive capacity
- Larger memory footprint
- Computational overhead

The conventional wisdom holds that returns diminish quickly past 1024-2048 dimensions. Our experiments challenge this.

## Experimental Setup

We trained a series of embedding models with identical architectures except for output dimension:

| Model | Dimensions | Parameters |
|-------|------------|------------|
| Zen-Embed-S | 768 | 110M |
| Zen-Embed-M | 1536 | 125M |
| Zen-Embed-L | 3072 | 155M |
| Zen-Embed-XL | 7680 | 230M |

Training data: 1.2B text pairs with contrastive learning objective.

## Results

### Retrieval Benchmarks

BEIR (Benchmarking IR) results across 15 datasets:

| Model | NDCG@10 | Recall@100 | MRR |
|-------|---------|------------|-----|
| Zen-Embed-S | 48.2 | 71.3 | 45.1 |
| Zen-Embed-M | 51.7 | 75.8 | 48.9 |
| Zen-Embed-L | 54.1 | 79.2 | 52.3 |
| Zen-Embed-XL | 57.3 | 83.6 | 55.8 |

The improvements continue well past conventional dimension counts.

### Scaling Analysis

Plotting performance against log(dimensions) reveals near-linear scaling:

$$\text{NDCG@10} \approx 0.12 \cdot \log_2(d) + 37.4$$

This suggests embedding capacity remains a bottleneck even at high dimensions.

### Per-Domain Breakdown

The benefits are not uniform across domains:

| Domain | 768d | 7680d | Improvement |
|--------|------|-------|-------------|
| Scientific | 42.1 | 54.7 | +30% |
| Legal | 38.9 | 51.2 | +32% |
| Conversational | 52.3 | 55.1 | +5% |
| News | 49.8 | 53.4 | +7% |

Technical and specialized domains benefit most. Everyday conversational content sees smaller gains.

### Interpretability

Higher dimensions don't just improve metrics; they enable finer distinctions. Analysis of the 7680d space shows:

- **Cleaner clusters**: Topic boundaries are sharper
- **Preserved nuance**: Similar but distinct concepts remain separable
- **Hierarchical structure**: Taxonomic relationships emerge naturally

## The Efficiency Question

7680 dimensions cost more to store and search. Is it worth it?

### Storage

| Dimensions | Bytes per Vector | 1M Vectors |
|------------|------------------|------------|
| 768 | 3,072 | 2.9 GB |
| 7680 | 30,720 | 29.3 GB |

10x storage for higher dimensions. Significant but manageable with modern hardware.

### Search Latency

Exact search scales linearly with dimensions. But approximate methods (HNSW, IVF) show sublinear scaling:

| Dimensions | Exact (ms) | HNSW (ms) | IVF-PQ (ms) |
|------------|------------|-----------|-------------|
| 768 | 12.3 | 0.8 | 0.3 |
| 7680 | 118.7 | 2.1 | 0.7 |

With appropriate indexing, 7680d search remains practical.

### Compression

Quantization recovers much of the efficiency loss:

- **INT8**: 4x compression, <1% quality loss
- **Binary**: 32x compression, 5% quality loss
- **Product Quantization**: 16x compression, 2% quality loss

## Practical Recommendations

Based on our experiments:

1. **If retrieval quality matters most**: Use 7680d with HNSW indexing
2. **If storage is constrained**: Use 7680d with INT8 quantization (still beats 768d float32)
3. **For conversational applications**: 1536d is sufficient
4. **For technical/specialized domains**: Higher dimensions provide outsized benefits

## Release

We're releasing the Zen-Embed family:

- **Zen-Embed-S** (768d): Free, MIT license
- **Zen-Embed-M** (1536d): Free, MIT license
- **Zen-Embed-L** (3072d): Free, MIT license
- **Zen-Embed-XL** (7680d): Free, MIT license

All models available on Hugging Face: huggingface.co/zoo-labs

## What This Means

The embedding dimension race isn't over. There's room to improve retrieval quality by increasing capacity. As hardware improves and indexing methods advance, higher-dimensional embeddings become increasingly practical.

More dimensions, better retrieval. Sometimes the simple approach works.

---

*Zach Kelling is a co-founder of Zoo Labs Foundation.*
