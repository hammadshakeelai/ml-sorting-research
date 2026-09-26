---
title: Map of Content - Machine Learning Sorting Research
tags:
  - moc
  - index
  - research
  - machine-learning
  - algorithms
created: 2026-09-26
---

# 🧠 Machine Learning Sorting Knowledge Base (Map of Content)

Welcome to the central index for research into **Machine Learning-Based Sorting, Complexity Limits, and Empirical Feasibility**. This vault documents theoretical proofs, neural architectures, learned algorithms, hardware constraints, benchmark logs, and future research roadmaps.

```
                            [[00_Map_of_Content]]
                                      |
       +------------------------------+------------------------------+
       |                              |                              |
       v                              v                              v
[[00_Roadmap/Planning]]     [[01_Theory/Limits]]         [[06_Repo/Layout]]
       |                              |                              |
       +------------------------------+------------------------------+
                                      |
       +------------------------------+------------------------------+
       |                                                             |
       v                                                             v
[[02_Models/Cpp_LearnedSort]]                             [[03_Hardware/Limits]]
[[02_Models/Neural_CDF]]                                             |
[[02_Models/LearnedSort]]                                            v
[[02_Models/Neural_Permutation]]                          [[04_Benchmarks/Scaling]]
       |                                                  [[04_Benchmarks/Multi_Dist]]
       +------------------------------+------------------------------+
                                      |
                                      v
                         [[05_Papers/Bibliography]]
```

---

## 📚 Vault Navigation & Index

### 0. Full Planning & Roadmap
* [[00_Research_Plan_and_Roadmap]]
  * The complete phase-by-phase research execution history: from theoretical bounds to neural permutations, monotonic CDFs, C++ production systems, and multi-distribution stress testing.
  * Future research horizons (AVX-512 vectorization, GPU CUDA kernels, distributed out-of-core sorting).

### 1. Theoretical Foundations & Complexity Bounds
* [[01_Theoretical_Foundations/Complexity_and_Limits]]
  * Why sequential sorting has an absolute lower bound of $\Omega(n)$.
  * The Information-Theoretic Comparison Bound: $\Omega(n \log n)$.
  * Formal Proof: Why sorting is strictly non-linear and static linear weights $W \cdot x$ cannot sort.
  * The Permutation Matrix Formulation: $y = P(x) \cdot x$.

### 2. Machine Learning & Neural Sorting Architectures
* [[02_ML_Architectures/Cpp_LearnedSort_Implementation]]  *(⭐ High-Performance C++ Milestone)*
  * High-performance C++ implementation compiled with `-O3 -march=native`.
  * **Beats GCC `std::sort` by 25% to 35%** across arrays from $n = 50,000$ to $3,000,000$.
  * Two-pass contiguous histogram buffering and L1-cache quantile spline dispatch.
* [[02_ML_Architectures/LearnedSort_CDF_Models]]
  * Breaking the comparison bound via Cumulative Distribution Function (CDF) estimation.
  * The $O(n)$ expected time proof: $\text{Rank}(x_i) \approx \lfloor (n - 1) \cdot \hat{F}(x_i) \rfloor$.
  * Recursive Model Indexes (RMI) and Monotonic Splines.
* [[02_ML_Architectures/Neural_CDF_and_Attention_Sort]]
  * **Deep Monotonic Neural CDF Sorter** (constrained positive weights via softplus for guaranteed monotonic CDF inference).
  * **Transformer Self-Attention Sorting Network** (evaluating multi-head query-key ranking).
* [[02_ML_Architectures/Neural_Permutation_Networks]]
  * Continuous relaxations of permutations (Gumbel-Sinkhorn Networks).
  * Why deep neural networks yield soft weights, inversions, and $O(n^2)$ scaling bottlenecks.

### 3. Hardware, Physical & Circuit Limits
* [[03_Hardware_and_Physics/Analog_Optical_Crossbars]]
  * Can hardware do $O(1)$ sorting?
  * Analog crossbar arrays (Memristors, Ohm's/Kirchhoff's Laws) and why dynamic permutations $P(x)$ prevent $O(1)$ sorting.
  * Circuit depth bounds in parallel architectures (AKS sorting networks, Bitonic networks).

### 4. Empirical Benchmarks & Experiments
* [[04_Empirical_Benchmarks/Benchmark_Logs_and_Scaling]]
  * Full benchmark data tables in Python and C++ ($n = 5$ to $3,000,000$).
  * Empirical scaling exponent fitting ($T(n) \sim n^b$).
* [[04_Empirical_Benchmarks/Multi_Distribution_Stress_Test]]
  * Robustness stress test across **6 real-world distributions** (Uniform, Gaussian, Exponential Skewed, Bimodal Mixture, Heavy Duplicates, Adversarial).
  * Verification of 100% exact mathematical sortedness.

### 5. Literature & Research Papers
* [[05_Literature_Review/Annotated_Bibliography]]
  * Key papers: Kraska et al. (SIGMOD 2020), DeepMind AlphaDev (Nature 2023), Grover et al. NeuralSort (ICLR 2019), Vinyals Pointer Networks.

### 6. Repository Structure & Guide
* [[06_Repository_Structure_and_Guide/Project_Layout]]
  * File-by-file directory map, build instructions, and developer documentation.

---

## 🔬 Scripts, Source Code & Executables
* C++ High-Performance Sorter: [`learned_sort.cpp`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/learned_sort.cpp)
* Advanced Multi-Model Suite: [`advanced_models_benchmark.py`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/advanced_models_benchmark.py)
* Multi-Distribution Benchmark JSON: [`multi_dist_benchmark_results.json`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/multi_dist_benchmark_results.json)
* Initial Benchmark Suite: [`benchmark_sorting_ml.py`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/benchmark_sorting_ml.py)
* Raw Benchmark JSON: [`benchmark_results.json`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/benchmark_results.json)
* Summary Research Report: [`research_report.md`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/research_report.md)
