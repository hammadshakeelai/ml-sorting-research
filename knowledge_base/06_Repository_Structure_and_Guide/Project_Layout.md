---
title: Repository Architecture, Directory Structure & Guide
tags:
  - repository
  - architecture
  - documentation
  - guide
created: 2026-09-26
parent: "[[00_Map_of_Content]]"
---

# 📁 Repository Architecture & Directory Guide

This document explains the organization of the codebase, the responsibilities of each file, and how researchers and developers can navigate and extend the project.

---

## 🌳 Directory Tree

```
mysterious-newton/
│
├── .gitignore                                <--- Git ignore rules for C++, Python, Obsidian
├── CMakeLists.txt                            <--- Cross-platform CMake build configuration
├── LICENSE                                   <--- MIT License (Hammad Shakeel)
├── README.md                                 <--- Comprehensive GitHub project front-page
├── research_report.md                        <--- Complete formal research paper & analysis
│
├── learned_sort.cpp                          <--- High-performance C++20 O(n) ML Sorter
│
├── benchmark_sorting_ml.py                   <--- Phase 1 Python benchmark suite
├── benchmark_results.json                    <--- Raw JSON logs from Phase 1
│
├── advanced_models_benchmark.py              <--- Phase 2 PyTorch multi-model suite
├── multi_dist_benchmark_results.json         <--- Raw JSON logs from Phase 2 stress-tests
│
└── knowledge_base/                           <--- STANDALONE OBSIDIAN VAULT
    ├── 00_Map_of_Content.md                  <--- Root Index & Obsidian Graph Hub
    ├── 00_Research_Plan_and_Roadmap.md       <--- Full planning, milestones & future horizons
    │
    ├── 01_Theoretical_Foundations/
    │   └── Complexity_and_Limits.md          <--- Mathematical proofs (Ω(n), non-linearity)
    │
    ├── 02_ML_Architectures/
    │   ├── Cpp_LearnedSort_Implementation.md <--- High-perf C++ implementation & memory layout
    │   ├── LearnedSort_CDF_Models.md         <--- O(n) distribution-aware CDF formulation
    │   ├── Neural_CDF_and_Attention_Sort.md  <--- Monotonic Softplus NN & Transformers
    │   └── Neural_Permutation_Networks.md    <--- Sinkhorn continuous relaxations & bottlenecks
    │
    ├── 03_Hardware_and_Physics/
    │   └── Analog_Optical_Crossbars.md       <--- Memristors, circuit depth, physical O(1) limits
    │
    ├── 04_Empirical_Benchmarks/
    │   ├── Benchmark_Logs_and_Scaling.md     <--- Complete empirical tables (N=5 to 3M)
    │   └── Multi_Distribution_Stress_Test.md <--- 6-distribution stress test analysis
    │
    ├── 05_Literature_Review/
    │   └── Annotated_Bibliography.md         <--- Landmark papers (MIT, DeepMind, Stanford)
    │
    └── 06_Repository_Structure_and_Guide/
        └── Project_Layout.md                 <--- This file
```

---

## 🛠️ File Descriptions & Responsibilities

### 1. High-Performance C++ Implementation
* **[`learned_sort.cpp`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/learned_sort.cpp)**:
  * Implements `LearnedSort`:
    * Step 1: Subsamples array and fits a 64-knot empirical CDF spline table.
    * Step 2: First $O(n)$ pass: generates histogram bucket counts.
    * Step 3: Computes prefix sums for flat contiguous scattering.
    * Step 4: Second $O(n)$ pass: scatters numbers into a flat buffer without dynamic heap allocations.
    * Step 5: Third $O(n)$ pass: in-place cache-friendly insertion sort per bucket.
  * Contains a benchmarking harness testing against `std::sort` (Introsort), `std::stable_sort`, and 64-bit LSD RadixSort on arrays up to 3,000,000 doubles.

### 2. Python & PyTorch Benchmark Suites
* **[`benchmark_sorting_ml.py`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/benchmark_sorting_ml.py)**:
  * Contains `NeuralPermutationSorter`: PyTorch module evaluating Gumbel-Sinkhorn iterations to learn $y = P(x) \cdot x$.
  * Contains pure Python `LearnedSort`, `pure_quicksort`, `radix_sort_uint`, and curve fitting for empirical exponent $b$ in $T(n) \sim n^b$.
* **[`advanced_models_benchmark.py`](file:///c:/Users/HP/Documents/antigravity/mysterious-newton/advanced_models_benchmark.py)**:
  * Contains `NeuralCDFNet`: Deep monotonic neural network with positive weights via $\text{Softplus}(W) > 0$.
  * Contains `TransformerAttentionSorter`: Transformer encoder analyzing query-key self-attention matrices.
  * Contains `RobustLearnedSort`: Multi-distribution stress tester evaluating Uniform, Gaussian, Skewed Exponential, Bimodal, Heavy Duplicates, and Nearly Sorted inputs.

### 3. Obsidian Vault (`knowledge_base/`)
* Open the `knowledge_base/` folder directly in [Obsidian](https://obsidian.md).
* Click on `00_Map_of_Content.md` or open the **Graph View** (`Ctrl+G`) to see all notes, mathematical proofs, and benchmarks visually connected via bidirectional `[[wikilinks]]`.
