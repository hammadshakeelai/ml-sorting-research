---
title: Deep Monotonic Neural CDF & Transformer Attention Sorting
tags:
  - machine-learning
  - neural-networks
  - transformer
  - attention
  - monotonic-nn
created: 2026-09-26
parent: "[[00_Map_of_Content]]"
see_also: "[[02_ML_Architectures/LearnedSort_CDF_Models]]"
---

# 🤖 Deep Monotonic Neural CDF & Transformer Attention Sorting

This note investigates two deep learning formulations for sorting:
1. **Deep Monotonic Neural CDF Network** (constraining weights to be strictly positive to learn a valid CDF).
2. **Transformer Multihead Self-Attention Network** (evaluating whether attention matrices discover permutation matrices).

---

## 1. Deep Monotonic Neural CDF Sorter

### Theoretical Constraint
A valid Cumulative Distribution Function $F(x) = P(X \le x)$ must satisfy two strict properties:
1. **Monotonic Non-Decreasing:** $x_1 \le x_2 \implies F(x_1) \le F(x_2)$.
2. **Bounded Output:** $\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to \infty} F(x) = 1$.

Standard neural networks can fluctuate non-monotonically, which breaks sorting. To guarantee monotonicity mathematically, we constrain all layer weights using the **Softplus operator**:
$$W_{ij} = \ln(1 + \exp(\theta_{ij})) > 0$$

```
Input x (1D)
    |
    v
+------------------------------------------+
|  Monotonic Dense Layer (W > 0) + ReLU    |
+------------------------------------------+
    |
    v
+------------------------------------------+
|  Monotonic Dense Layer (W > 0) + ReLU    |
+------------------------------------------+
    |
    v
+------------------------------------------+
|  Monotonic Dense Layer (W > 0) + Sigmoid |
+------------------------------------------+
    |
    v
Output: F_theta(x) in (0, 1) strictly monotonic!
```

### Empirical Results (Evaluated on $n = 2,000$):
* **Training Time:** $0.50\text{ s}$ for 250 epochs.
* **Inference & Sorting Latency:** $3.26\text{ ms}$.
* **Exact Mathematical Sortedness:** **100% (True)**.
* **Takeaway:** When neural networks are used as **functional CDF approximators** combined with bucket dispatching, they successfully achieve **$O(n)$ exact sorting**!

---

## 2. Transformer Multihead Self-Attention Sorting

Can a standard Transformer Encoder learn to sort sequences via self-attention without hand-crafted heuristics?

### Architecture:
* Input: Unsorted token sequence $x = [x_1, \dots, x_L]$.
* Query-Key Dot Product Attention:
  $$A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{L \times L}$$
* Feed-forward projection to target sorted sequence.

### Empirical Observation:
* **Input Vector:** `[0.395, 0.163, 0.557, 0.399, 0.348, 0.975, 0.189, 0.135]`
* **Target Sorted:** `[0.135, 0.163, 0.189, 0.348, 0.395, 0.399, 0.557, 0.975]`
* **Transformer Prediction:** `[0.112, 0.196, 0.261, 0.317, 0.423, 0.596, 0.773, 1.038]`

### Analysis:
1. The Transformer successfully learned the ascending ranking pattern (smooth monotonic curve).
2. However, pure attention outputs are **continuous regressions**. It does not perform exact permutation (it produces continuous interpolated values rather than reordering the original discrete tokens).
3. The computational complexity of full self-attention is $O(L^2 \cdot d)$, making it asymptotically inferior to $O(n \log n)$ or $O(n)$ sorting for large $n$.
