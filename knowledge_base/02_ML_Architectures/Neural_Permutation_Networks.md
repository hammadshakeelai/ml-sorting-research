---
title: Neural Permutation Networks & Continuous Relaxations
tags:
  - machine-learning
  - neural-networks
  - sinkhorn
  - differentiability
created: 2026-09-26
parent: "[[00_Map_of_Content]]"
see_also: "[[01_Theoretical_Foundations/Complexity_and_Limits]]"
---

# 🧠 Neural Permutation Networks (Sinkhorn-Knopp)

## 1. Formulation
A discrete permutation matrix $P \in \{0, 1\}^{n \times n}$ has discrete entries:
$$\sum_{j=1}^n P_{ij} = 1 \quad \forall i, \qquad \sum_{i=1}^n P_{ij} = 1 \quad \forall j$$
Standard gradient descent cannot propagate gradients through discrete permutation indices.

To make sorting differentiable in deep neural networks, researchers (e.g. Mena et al., 2018; Grover et al., 2019) use **continuous relaxations into doubly-stochastic matrices**:
$$P_{\tau}(x) = \text{Sinkhorn}\left(\frac{S(x)}{\tau}\right)$$
where $S(x)$ is a pairwise score matrix and $\tau > 0$ is a temperature parameter.

---

## 2. The Sinkhorn-Knopp Algorithm
Given an unconstrained positive matrix $A = \exp(S / \tau)$:
1. Normalize rows: $A \leftarrow D_r^{-1} A$
2. Normalize columns: $A \leftarrow A D_c^{-1}$
3. Repeat for $L$ iterations. By the Sinkhorn-Knopp theorem, this converges to a doubly-stochastic matrix where each entry represents the probability that element $x_j$ belongs at rank $i$.

The sorted output vector is obtained via batch matrix multiplication:
$$y_{\text{sorted}} = P_{\tau}(x)^T \cdot x$$

---

## 3. Why Deep Neural Sorting Fails for Production
Our empirical experiments ([`benchmark_results.json`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/benchmark_results.json)) reveal three critical failure modes:

| Metric | Neural Permutation Net | Classical Sort |
| :--- | :--- | :--- |
| **Exact Mathematical Sort** | **0.0%** | **100.0%** |
| **Time Complexity** | $O(n^2)$ to $O(n^3)$ | $O(n \log n)$ / $O(n)$ |
| **Space Complexity** | $O(n^2)$ matrix memory | $O(1)$ to $O(n)$ |
| **Latency ($n = 50$)** | $\sim 1.96\text{ ms}$ | $\sim 0.005\text{ ms}$ ($400\times$ faster) |

### Key Reasons:
1. **Continuous Softness vs. Strict Sorting**:
   Even if the network achieves a low MSE loss ($\sim 0.001$), rounding real-valued scores introduces duplicate assignments or subtle inversions. A single inverted pair means the array is technically unsorted.
2. **Quadratic Memory Scaling**:
   Allocating an $n \times n$ matrix for $n = 100,000$ requires:
   $$100,000^2 \times 4\text{ bytes} = 40\text{ Gigabytes}$$
   This causes out-of-memory (OOM) errors on modern GPUs/CPUs.
3. **Purpose of NeuralSort**:
   NeuralSort was designed for **end-to-end backpropagation** (e.g., top-$k$ supervision, ranking supervision in recommenders), NOT for raw wall-clock algorithmic sorting speed.
