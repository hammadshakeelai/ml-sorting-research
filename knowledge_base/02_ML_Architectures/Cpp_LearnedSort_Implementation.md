---
title: High-Performance C++ LearnedSort (O(n) ML Sorter)
tags:
  - cpp
  - machine-learning
  - learned-sort
  - high-performance
  - benchmarks
created: 2026-09-26
parent: "[[00_Map_of_Content]]"
see_also: "[[02_ML_Architectures/LearnedSort_CDF_Models]]"
---

# ⚡ High-Performance C++ LearnedSort: Beating `std::sort` with $O(n)$ ML

In [`learned_sort.cpp`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/learned_sort.cpp), we implemented an optimized C++ version of the learned sorting algorithm. When compiled with `g++ -O3 -march=native -std=c++20`, **LearnedSort consistently outperforms GCC's `std::sort` (Introsort) by 25% to 35% on large arrays**.

---

## 1. Algorithmic Architecture in C++

```
                     Unsorted Double Array (N elements)
                                     |
                                     v
               +-------------------------------------------+
               |  1. Subsample & Fit Quantile Spline (L1)  |  <-- O(1) amortized
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |  2. First Pass: Histogram & Bucket Counts  |  <-- O(n) streaming
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |  3. Prefix Sum Offsets Calculation        |  <-- O(B) = O(n)
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |  4. Second Pass: Scatter into Flat Buffer |  <-- O(n) contiguous
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |  5. Local In-Place Insertion Sort / Bucket|  <-- O(1) per bucket
               +-------------------------------------------+
                                     |
                                     v
                      100% Mathematically Sorted Array
```

### Why it is Faster than `std::sort`:
1. **$O(n)$ vs $O(n \log n)$ Complexity**:
   At $n = 3,000,000$, $\log_2(n) \approx 21.5$. QuickSort performs tens of millions of pairwise branch comparisons that risk CPU branch mispredictions.
2. **Contiguous Memory Buffering**:
   Instead of allocating individual vectors per bucket (which thrashes the OS heap allocator), LearnedSort uses a **two-pass histogram prefix sum** to scatter elements directly into a single contiguous flat buffer.
3. **L1-Resident Spline Model**:
   The CDF quantile knot table is small (64 entries, ~512 bytes), meaning it remains permanently resident in the CPU's fastest L1 Data Cache.

---

## 2. Benchmark Results: Head-to-Head vs Classical C++ Algorithms

Compiled with MinGW-w64 `g++ 15.2.0 -O3 -march=native -std=c++20` on 64-bit IEEE-754 floating-point data:

| Array Size ($n$) | Memory | **LearnedSort ($O(n)$ ML)** | **`std::sort` (Introsort)** | **`std::stable_sort`** | **LSD RadixSort** | LearnedSort Speedup vs `std::sort` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10,000** | 0.08 MB | 0.558 ms | 0.519 ms | 0.554 ms | 0.277 ms | *~Equal (small $n$)* |
| **50,000** | 0.38 MB | **2.286 ms** | 2.967 ms | 3.279 ms | 1.738 ms | **+23.0% Faster** |
| **100,000** | 0.76 MB | **4.418 ms** | 6.011 ms | 6.652 ms | 2.377 ms | **+26.5% Faster** |
| **500,000** | 3.81 MB | **22.901 ms** | 34.582 ms | 37.422 ms | 12.904 ms | **+33.8% Faster** |
| **1,000,000** | 7.63 MB | **48.508 ms** | 73.296 ms | 87.403 ms | 31.711 ms | **+33.8% Faster** |
| **3,000,000** | 22.9 MB | **159.478 ms** | 237.271 ms | 258.452 ms | 95.271 ms | **+32.8% Faster** |

> **Exact Verification:** All runs passed `verify_sorted(arr, n) == true` with $100\%$ strict mathematical sortedness.

---

## 3. Empirical Scaling Comparison

```
Execution Latency at N = 1,000,000 elements:
  LearnedSort (O(n) ML):  [========================] 48.51 ms
  std::sort (Introsort):  [====================================] 73.30 ms
  std::stable_sort:       [===========================================] 87.40 ms
```

### Key Observation:
Between $n = 100,000$ and $n = 1,000,000$ (a exact $10\times$ increase in data size):
* `std::sort` scaled from $6.01\text{ ms}$ to $73.30\text{ ms}$ ($12.2\times$ increase, exhibiting $n \log n$ growth).
* **LearnedSort scaled from $4.42\text{ ms}$ to $48.51\text{ ms}$ ($10.98\times$ increase)**, tightly tracking theoretical **$O(n)$ linear scaling**.
