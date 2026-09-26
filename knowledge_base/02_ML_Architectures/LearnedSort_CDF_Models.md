---
title: LearnedSort & CDF-Based Machine Learning Sorting
tags:
  - machine-learning
  - learned-sort
  - linear-time
  - cdf
created: 2026-09-26
parent: "[[00_Map_of_Content]]"
see_also: "[[01_Theoretical_Foundations/Complexity_and_Limits]]"
---

# 🚀 LearnedSort & CDF-Based Sorting

## 1. The Core Breakthrough
Introduced by Tim Kraska (MIT) and Ani Kristo (Brown) in *SIGMOD 2020*, **LearnedSort** reframes sorting from **comparing pairs** to **learning the data distribution**.

If an array $X = [x_1, \dots, x_n]$ is drawn from a continuous probability distribution with Cumulative Distribution Function:
$$F(x) = P(X \le x)$$
the expected rank of any element $x_i$ in the sorted array is:
$$\mathbb{E}[\text{Rank}(x_i)] = (n - 1) \cdot F(x_i)$$

```
        Raw Data Point: x_i
                 |
                 v
   +-----------------------------+
   |   Learned Model / Spline    |   <-- Evaluated in O(1) time
   |      F_hat(x) in [0, 1]     |
   +-----------------------------+
                 |
                 v
        Target Bucket Index:
    floor((num_buckets - 1) * F_hat(x_i))
                 |
                 v
   +-----------------------------+
   |  Direct Bucket Insertion    |   <-- O(1) insertion
   +-----------------------------+
                 |
                 v
   +-----------------------------+
   |  Local Resolution (O(1))    |   <-- Cache-friendly local sort
   +-----------------------------+
```

---

## 2. Algorithm Steps & Runtime Breakdown

1. **Subsampling Step ($O(s \log s)$ where $s \ll n$)**:
   Take a small random sample of size $s = \sqrt{n}$ or $s = \min(2048, 0.05n)$. Sort the sample.
2. **Model Fitting ($O(s)$)**:
   Fit a monotonic piecewise linear spline or lightweight 2-layer monotonic model approximating the empirical CDF $\hat{F}(x)$.
3. **Partitioning Pass ($O(n)$)**:
   Iterate over all $n$ items. For each $x_i$:
   $$\text{bucket} = \lfloor (B - 1) \cdot \hat{F}(x_i) \rfloor$$
   Place $x_i$ into bucket $B[\text{bucket}]$. Total time: $n \times O(1) = O(n)$.
4. **Local Sort Pass ($O(n)$ expected)**:
   Sort each bucket using Insertion Sort. If the model error is bounded, each bucket contains an average of $O(1)$ items. Insertion sort on $k$ items takes $O(k^2)$, so $\sum O(k^2) = O(n)$.
5. **Concatenation ($O(n)$)**:
   Concatenate all non-empty buckets sequentially.

$$\mathbf{T_{\text{total}}(n) = O(s \log s) + O(n) + O(n) = O(n)}$$

---

## 3. Why LearnedSort Beats RadixSort
Traditional non-comparison sorting (like RadixSort) operates on byte passes:
$$T_{\text{Radix}}(n) = O\left(\frac{w}{b} \cdot (n + 2^b)\right)$$
For 64-bit floating-point numbers ($w = 64$), RadixSort requires multiple cache-thrashing passes (often 4 to 8 passes over memory).

In contrast, LearnedSort:
* Makes **a single linear pass** over the array to place items into cache-local buckets.
* Handles arbitrary non-integer continuous data, skewed distributions, and heavy tails naturally.
* Achieves **$100\%$ exact sorting** because the bucket resolution step guarantees strict mathematical order.
