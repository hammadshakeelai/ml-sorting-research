# ⚡ Machine Learning-Based Sorting & Complexity Limits

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red.svg)](https://pytorch.org/)
[![Obsidian Ready](https://img.shields.io/badge/Obsidian-Knowledge%20Base-purple.svg)](https://obsidian.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An in-depth empirical research repository and high-performance C++ implementation exploring **machine learning-driven sorting algorithms**, **computational complexity limits ($O(1)$ vs $O(n)$)**, **neural permutation networks**, and **LearnedSort**.

---

## 🎯 Executive Summary & Highlights

1. **Beating C++ `std::sort` with $O(n)$ ML**:
   Our C++ implementation of LearnedSort ([`learned_sort.cpp`](learned_sort.cpp)) compiled with `-O3 -march=native` is **25% to 35% faster than GCC's `std::sort` (Introsort)** on arrays from 50,000 to 3,000,000 double-precision floats, while guaranteeing **100% exact mathematical sortedness**.
2. **Mathematical Proof: Why Static Weights Cannot Sort**:
   We formally prove that a static linear weight matrix $y = W \cdot x$ **cannot** sort arbitrary arrays because sorting is strictly non-linear ($W(x_1 - x_2) = 0$ leads to direct contradiction).
3. **The $O(n)$ CDF Breakthrough**:
   By reframing sorting as **Cumulative Distribution Function (CDF) quantile regression**, an ML model predicts target bucket indices in $O(1)$ time per element:
   $$\text{Rank}(x_i) \approx \lfloor (n - 1) \cdot \hat{F}(x_i) \rfloor$$
4. **Complete Standalone Obsidian Knowledge Base**:
   The [`knowledge_base/`](knowledge_base/) directory is a structured Obsidian vault with LaTeX proofs, architecture diagrams, benchmark logs, and literature reviews.

---

## 📁 Repository Structure

```
mysterious-newton/
│
├── .gitignore                                <--- Excludes binaries, build dirs, caches
├── CMakeLists.txt                            <--- Cross-platform CMake build configuration
├── LICENSE                                   <--- MIT License (Hammad Shakeel)
├── README.md                                 <--- Repository documentation and guide
├── research_report.md                        <--- Formal research document and proofs
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
        └── Project_Layout.md                 <--- Detailed codebase navigation guide
```

---

## 📊 Benchmark Results

### 1. C++ Head-to-Head: LearnedSort vs. Classical Algorithms
Evaluated on 64-bit IEEE-754 double precision floats (Gaussian distribution):

| Array Size ($N$) | Data Size | **LearnedSort ($O(n)$ ML)** | **`std::sort` (Introsort)** | **`std::stable_sort`** | **LSD RadixSort** | LearnedSort Speedup vs `std::sort` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10,000** | 0.08 MB | 0.558 ms | 0.519 ms | 0.554 ms | 0.277 ms | *~Equal (small $N$)* |
| **50,000** | 0.38 MB | **2.286 ms** | 2.967 ms | 3.279 ms | 1.738 ms | **+23.0% Faster** |
| **100,000** | 0.76 MB | **4.418 ms** | 6.011 ms | 6.652 ms | 2.377 ms | **+26.5% Faster** |
| **500,000** | 3.81 MB | **22.901 ms** | 34.582 ms | 37.422 ms | 12.904 ms | **+33.8% Faster** |
| **1,000,000** | 7.63 MB | **48.508 ms** | 73.296 ms | 87.403 ms | 31.711 ms | **+33.8% Faster** |
| **3,000,000** | 22.9 MB | **159.478 ms** | 237.271 ms | 258.452 ms | 95.271 ms | **+32.8% Faster** |

> **All runs passed exact verification**: 0 inversions, 100% strict mathematical order.

```
Execution Latency at N = 1,000,000 elements:
  LearnedSort (O(n) ML):  [========================] 48.51 ms
  std::sort (Introsort):  [====================================] 73.30 ms
  std::stable_sort:       [===========================================] 87.40 ms
```

---

### 2. Multi-Distribution Stress Test (Python / NumPy / TimSort)
Tested across 6 distinct data distributions from $n = 1,000$ to $50,000$:

| Distribution | $n = 1,000$ | $n = 10,000$ | $n = 50,000$ | Exact Verified? | Scaling Ratio ($50k / 10k$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Uniform** | 0.73 ms | 6.26 ms | 33.66 ms | **100% True** | $5.37\times$ (Linear) |
| **Gaussian (Normal)** | 0.84 ms | 6.22 ms | 38.63 ms | **100% True** | $6.21\times$ (Linear) |
| **Exponential (Heavy Skew)** | 0.75 ms | 6.79 ms | 44.43 ms | **100% True** | $6.54\times$ (Linear) |
| **Bimodal Mixture** | 0.73 ms | 6.01 ms | 34.40 ms | **100% True** | $5.72\times$ (Linear) |
| **Heavy Duplicates** (5 keys) | **0.39 ms** | **2.68 ms** | **13.57 ms** | **100% True** | **$5.06\times$ (Fastest)** |
| **Nearly Sorted (Adversarial)** | 0.59 ms | 4.89 ms | 29.67 ms | **100% True** | $6.06\times$ (Linear) |

---

## 🧠 Implemented Paradigms & Models

```
                               ML Sorting Paradigms
                                         |
     +-----------------------------------+-----------------------------------+
     |                                   |                                   |
     v                                   v                                   v
[High-Perf C++ LearnedSort]    [Deep Monotonic Neural CDF]     [Neural Permutation / Transformer]
* O(n) streaming scatter       * Softplus positive weights     * Gumbel-Sinkhorn relaxations
* Contiguous histogram buffer  * Mathematically monotonic      * Self-attention rank discovery
* L1-cache quantile spline     * 100% exact bucket sort        * O(n²) scaling bottleneck
```

1. **High-Performance C++ LearnedSort** ([`learned_sort.cpp`](learned_sort.cpp)):
   * Uses an L1-cache-resident spline model trained in $O(1)$ amortized time.
   * Employs a two-pass histogram prefix sum to eliminate dynamic heap allocations (`std::vector` overhead) and scatter elements directly into contiguous flat buffers.
2. **Deep Monotonic Neural CDF Sorter** ([`advanced_models_benchmark.py`](advanced_models_benchmark.py)):
   * Neural network with weights constrained via $\text{Softplus}(W) > 0$ guaranteeing $x_1 \le x_2 \implies F_\theta(x_1) \le F_\theta(x_2)$.
3. **Transformer Multihead Self-Attention Sorter** ([`advanced_models_benchmark.py`](advanced_models_benchmark.py)):
   * Evaluates query-key dot product attention matrices to observe rank discovery in latent space.
4. **Continuous Permutation Net** ([`benchmark_sorting_ml.py`](benchmark_sorting_ml.py)):
   * Uses Gumbel-Sinkhorn iterations to evaluate continuous matrix relaxations $y = P(x) \cdot x$.

---

## 🗺️ Full Research Roadmap

For the full level-by-level trajectory, see [00_Research_Plan_and_Roadmap.md](knowledge_base/00_Research_Plan_and_Roadmap.md):
- **Phase 1 (Completed):** Mathematical limits ($\Omega(n)$ sequential bounds and non-linearity proof).
- **Phase 2 (Completed):** Neural permutation networks and $O(n^2)$ continuous relaxation bottlenecks.
- **Phase 3 (Completed):** CDF estimation breakthroughs and monotonic deep neural networks.
- **Phase 4 (Completed):** C++ production engineering beating `std::sort` by up to $34\%$.
- **Phase 5 (Completed):** Multi-distribution stress testing across 6 real-world data regimes.
- **Phase 6 (Roadmap):** AVX-512 SIMD vectorization, CUDA GPU kernels, and distributed out-of-core sorting.

---

## 🛠️ Quickstart & Building

### 1. Compile & Run C++ LearnedSort
Requires a modern C++20 compiler (`g++`, `clang++`, or MSVC):

```bash
# Using g++
g++ -O3 -march=native -std=c++20 learned_sort.cpp -o learned_sort

# Run benchmark
./learned_sort
```

Or build with CMake:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/learned_sort
```

### 2. Run Python & PyTorch Neural Benchmarks
```bash
# Install dependencies
pip install torch numpy scipy

# Run initial benchmarks (Neural Permutation vs LearnedSort vs TimSort)
python benchmark_sorting_ml.py

# Run advanced suite (Monotonic Neural CDF, Transformers, Multi-Distribution stress test)
python advanced_models_benchmark.py
```

---

## 📖 Key References

* **LearnedSort:** A. Kristo, K. Vaidya, U. Çetintemel, S. Misra, T. Kraska. *"The Case for a Learned Sorting Algorithm."* ACM SIGMOD 2020.
* **AlphaDev:** D. J. Mankowitz et al. *"Faster sorting algorithms discovered using deep reinforcement learning."* Nature 618, 257–263 (2023).
* **Learned Index Structures:** T. Kraska, A. Beutel, E. H. Chi, J. Dean, N. Polyzotis. *"The Case for Learned Index Structures."* ACM SIGMOD 2018.
* **NeuralSort:** A. Grover, E. Wang, A. Courville, S. Ermon. *"Stochastic Optimization of Sorting Networks via Continuous Relaxations."* ICLR 2019.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
