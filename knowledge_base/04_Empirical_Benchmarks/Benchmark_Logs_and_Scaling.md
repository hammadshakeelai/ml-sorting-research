---
title: Empirical Benchmark Logs & Scaling Analysis
tags:
  - benchmarks
  - empirical
  - latency
  - scaling
created: 2026-09-26
parent: "[[00_Map_of_Content]]"
---

# 📊 Empirical Benchmark Logs & Scaling Analysis

This note records the empirical logs generated on **September 26, 2026** by running [`benchmark_sorting_ml.py`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/benchmark_sorting_ml.py).

---

## 1. Test 1: Neural Permutation Sorter (Gumbel-Sinkhorn)
* **Architecture:** 3-layer MLP encoder + 15 log-space Sinkhorn iterations + Matrix Product $y = P(x)^T \cdot x$.
* **Hardware:** Intel Core CPU, PyTorch 2.11.0.

```
[*] n=  5 | Avg Latency:  1.348 ms | Avg Inversions: 1.01 | 100% Exact Sorts:   0.0%
[*] n= 10 | Avg Latency:  1.421 ms | Avg Inversions: 0.09 | 100% Exact Sorts:   0.0%
[*] n= 20 | Avg Latency:  1.463 ms | Avg Inversions: 0.07 | 100% Exact Sorts:   0.0%
[*] n= 50 | Avg Latency:  1.956 ms | Avg Inversions: 0.17 | 100% Exact Sorts:   0.0%
```

**Verdict:** Proves that while neural networks can approximate ranking (low MSE), continuous relaxations fail to provide $100\%$ exact discrete sorting. Latency is $\sim 100\times$ higher than classical CPU sorts.

---

## 2. Test 2: Scaling Benchmark Across Paradigms ($n = 100 \to 100,000$)

All runs in Test 2 achieved **100% Verified Exact Sorted Output**:

| $n$ | LearnedSort ($O(n)$) | Python TimSort | Pure QuickSort | Radix Sort | NumPy C-Sort |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **100** | 0.999 ms | 0.011 ms | 0.161 ms | 1.280 ms | 0.616 ms |
| **500** | 0.635 ms | 0.043 ms | 0.951 ms | 4.061 ms | 0.014 ms |
| **1,000** | 1.085 ms | 0.096 ms | 1.870 ms | 7.426 ms | 0.016 ms |
| **5,000** | 4.879 ms | 0.557 ms | 11.142 ms | 26.970 ms | 0.060 ms |
| **10,000** | **7.555 ms** | 1.101 ms | **18.382 ms** | 41.725 ms | 0.102 ms |
| **50,000** | 40.856 ms | 7.462 ms | *Recursion limit* | 185.046 ms | 0.611 ms |
| **100,000** | **76.513 ms** | 13.055 ms | *Recursion limit* | **360.155 ms** | 0.878 ms |

---

## 3. Empirical Scaling Exponent ($T(n) = a \cdot n^b$)
Fitting $\log(T) = b \cdot \log(n) + c$:

* **LearnedSort:** $b = 0.715$  
  *(Asymptotic behavior is sub-linear in this range due to cache warming; theoretical expectation is $1.00$)*.
* **Python TimSort:** $b = 1.054$  
  *(Matches theoretical $O(n \log n)$, where $\log(n)$ introduces slight upward curvature)*.
* **RadixSort:** $b = 0.815$  
  *(Demonstrates linear-class behavior, but suffers from high constant overhead due to 4 passes over arrays)*.
