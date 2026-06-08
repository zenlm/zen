---
title: "Zen Reranker: Two-Stage Retrieval Done Right"
date: "2023-03-13"
description: "Introducing the Zen Reranker, a cross-encoder model that dramatically improves retrieval quality in two-stage pipelines."
author: "Zach Kelling"
tags: ["Research", "Retrieval", "Reranking"]
---


Embedding-based retrieval is fast but imprecise. Cross-encoder reranking is precise but slow. The combination unlocks the best of both. Today we release the Zen Reranker, purpose-built for two-stage retrieval.

## Two-Stage Retrieval

Modern retrieval pipelines typically operate in two stages:

```
Query -> [Embedding Retrieval] -> Top-K Candidates -> [Reranker] -> Final Results
         (fast, approximate)                          (slow, precise)
```

**Stage 1**: Bi-encoder embeddings enable fast approximate search over millions of documents. Retrieve top-100 to top-1000 candidates.

**Stage 2**: Cross-encoder reranker scores each candidate against the query with full attention. Reorder to get final top-10 or top-20.

The reranker sees query-document pairs together, enabling much finer relevance distinctions than independent embeddings.

## The Zen Reranker

Our reranker builds on several key design decisions:

### Architecture

- **Base**: 330M parameter encoder-only transformer
- **Input**: Concatenated query and document with separator tokens
- **Output**: Single relevance score (0-1)
- **Context**: 512 tokens (query + document combined)

### Training Data

We curated training data from multiple sources:

1. **MS MARCO**: Search query-passage pairs (positive and hard negatives)
2. **Natural Questions**: Question-answer pairs from Wikipedia
3. **Synthetic pairs**: LLM-generated query-document pairs with labels
4. **Human judgments**: 50K expert-annotated pairs across domains

Total: 12M training pairs with 4-way relevance labels.

### Training Objective

We use a listwise loss that considers the full ranking:

```python
def listwise_loss(scores, labels):
    # Softmax over candidate scores
    probs = softmax(scores)
    # Cross-entropy with label distribution
    return -sum(labels * log(probs))
```

This outperforms pointwise binary classification by teaching the model to rank, not just classify.

## Benchmark Results

### BEIR Reranking

Reranking BM25 top-100 results:

| Reranker | NDCG@10 | Time (ms/query) |
|----------|---------|-----------------|
| No reranking | 42.1 | - |
| monoT5-base | 49.3 | 180 |
| MiniLM-reranker | 47.8 | 45 |
| **Zen Reranker** | 52.7 | 62 |

### Reranking Zen Embeddings

Combined with our 7680d embeddings:

| Pipeline | NDCG@10 | Latency |
|----------|---------|---------|
| Zen-Embed only | 57.3 | 8ms |
| Zen-Embed + Reranker | 64.1 | 70ms |

The two-stage pipeline achieves +12% improvement with acceptable latency overhead.

### Domain-Specific Performance

| Domain | BM25 | +Zen Reranker | Improvement |
|--------|------|---------------|-------------|
| Scientific (SCIDOCS) | 15.8 | 21.4 | +35% |
| Finance (FiQA) | 29.6 | 38.2 | +29% |
| Covid (TREC-COVID) | 65.5 | 78.3 | +20% |
| Quora duplicate | 78.9 | 84.6 | +7% |

Specialized domains with complex language benefit most.

## Usage

### Basic Usage

```python
from zen.reranker import ZenReranker

reranker = ZenReranker.from_pretrained("zoo-labs/zen-reranker")

query = "What causes climate change?"
documents = [
    "Greenhouse gases trap heat in the atmosphere...",
    "The weather today is sunny and warm...",
    "CO2 emissions from burning fossil fuels..."
]

scores = reranker.score(query, documents)
# [0.92, 0.12, 0.87]
```

### Integration with Retrievers

```python
from zen.retrieval import ZenRetriever
from zen.reranker import ZenReranker

retriever = ZenRetriever("zoo-labs/zen-embed-xl")
reranker = ZenReranker.from_pretrained("zoo-labs/zen-reranker")

# Stage 1: Fast retrieval
candidates = retriever.retrieve(query, k=100)

# Stage 2: Precise reranking
reranked = reranker.rerank(query, candidates, k=10)
```

### Batched Inference

For production workloads:

```python
# Batch queries for efficiency
queries = ["query 1", "query 2", ...]
candidate_lists = [[docs...], [docs...], ...]

results = reranker.batch_rerank(
    queries, 
    candidate_lists,
    batch_size=32,
    k=10
)
```

## Deployment Considerations

### Hardware Requirements

- **Minimum**: 4GB GPU memory
- **Recommended**: 8GB+ for batched inference
- **CPU**: Viable for low-throughput (<10 QPS)

### Latency Optimization

1. **Batching**: Process multiple query-doc pairs together
2. **Quantization**: INT8 reduces latency 40% with <1% quality loss
3. **Early termination**: Stop scoring when top-k is confident
4. **Caching**: Cache scores for repeated query-document pairs

### Scaling

For high-throughput applications:
- Deploy multiple replicas behind load balancer
- Use async inference with request queuing
- Consider distilled smaller models for extreme latency requirements

## Model Release

The Zen Reranker is available under Apache 2.0:

- **Hugging Face**: huggingface.co/zoo-labs/zen-reranker
- **ONNX**: Optimized for deployment
- **TensorRT**: NVIDIA-optimized variant

## Conclusion

Two-stage retrieval with a quality reranker is the pragmatic choice for production search systems. The Zen Reranker provides state-of-the-art reranking in an efficient, easy-to-deploy package.

Fast first, then precise.

---

*Zach Kelling is a co-founder of Zoo Labs Foundation.*
