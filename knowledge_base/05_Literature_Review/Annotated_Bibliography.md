---
title: Annotated Bibliography & Research Papers
tags:
  - literature
  - papers
  - bibliography
  - citations
created: 2026-09-26
parent: "[[00_Map_of_Content]]"
---

# 📖 Annotated Bibliography & Landmark Research Papers

This note provides summaries, citations, and critical commentary on landmark papers at the intersection of **Machine Learning, Deep Learning, and Sorting Algorithms**.

---

## 1. The Learned Systems & Learned Sort Paradigm

### 📄 *"The Case for a Learned Sorting Algorithm"*
* **Authors:** Ani Kristo, Kapil Vaidya, Uğur Çetintemel, Sanchit Misra, Tim Kraska (MIT, Brown University, Intel)
* **Venue:** ACM SIGMOD 2020
* **DOI / Citation:** [arXiv:2005.10919](https://arxiv.org/abs/2005.10919)
* **Core Contribution:**
  * Replaces standard partitioning routines in QuickSort or RadixSort with **Cumulative Distribution Function (CDF)** models.
  * Shows that ML models can approximate the empirical distribution $F(x)$ to place items into their destination buckets in $O(1)$ time per element.
  * Demonstrates that in C++, LearnedSort is up to **$3.38\times$ faster than `std::sort`** and **$1.49\times$ faster than RadixSort**.

### 📄 *"The Case for Learned Index Structures"*
* **Authors:** Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean, Neoklis Polyzotis (Google Brain, MIT)
* **Venue:** ACM SIGMOD 2018
* **DOI / Citation:** [arXiv:1712.01208](https://arxiv.org/abs/1712.01208)
* **Core Contribution:**
  * The seminal paper establishing that core data structures (B-Trees, Hash Tables, Bloom Filters) can be viewed as functional approximations of CDFs.
  * Proves that neural networks / linear spline models can predict the exact memory position of keys faster and using up to $70\%$ less memory than standard B-Trees.

---

## 2. Deep Learning & Differentiable Sorting

### 📄 *"Faster sorting algorithms discovered using deep reinforcement learning" (AlphaDev)*
* **Authors:** Daniel J. Mankowitz et al. (Google DeepMind)
* **Venue:** *Nature*, 618, 257–263 (2023)
* **DOI / Citation:** [Nature: s41586-023-06004-9](https://doi.org/10.1038/s41586-023-06004-9)
* **Core Contribution:**
  * Uses Deep Reinforcement Learning (an AlphaZero-style MCTS agent) to play an assembly-code game ("AssemblyGame") searching for optimal x86-64 assembly instructions to sort short sequences ($n = 3, 4, 5$).
  * AlphaDev discovered novel branchless instructions that eliminated a swap operation, leading to algorithms up to **$70\%$ faster** for short sequences.
  * The code was officially merged into LLVM’s standard C++ library (`libc++`).

### 📄 *"Stochastic Optimization of Sorting Networks via Continuous Relaxations" (NeuralSort)*
* **Authors:** Aditya Grover, Eric Wang, Aaron Courville, Stefano Ermon (Stanford University, Mila)
* **Venue:** ICLR 2019
* **DOI / Citation:** [arXiv:1903.08850](https://arxiv.org/abs/1903.08850)
* **Core Contribution:**
  * Introduces **NeuralSort**, a continuous, differentiable relaxation of the sorting operator using the Plackett-Luce distribution and temperature-annealed softmax.
  * Enables gradients to flow through rank-based objective functions.
  * Computationally requires $O(n^2)$ pairwise comparisons; designed for training supervision, not algorithmic sorting speed.

### 📄 *"Learning Latent Permutations with Gumbel-Sinkhorn Networks"*
* **Authors:** Gonzalo Mena, David Belanger, Scott Linderman, Jasper Snoek (Harvard University, Google Brain)
* **Venue:** ICLR 2018
* **DOI / Citation:** [arXiv:1802.08665](https://arxiv.org/abs/1802.08665)
* **Core Contribution:**
  * Bridges discrete matching problems with gradient-based optimization using the **Sinkhorn-Knopp algorithm** applied to unnormalized log-potentials perturbed by Gumbel noise.
