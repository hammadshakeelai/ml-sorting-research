"""
Machine Learning Sorting Benchmark & Feasibility Suite
Explores:
1. Dynamic Weight Neural Permutation Net (PyTorch Sinkhorn-Knopp) - y = W(x) * x
2. O(n) LearnedSort (CDF Estimation + Bucketing + Local Sort)
3. Classical Baselines (QuickSort, RadixSort, TimSort, NumPy Introsort)
"""

import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =====================================================================
# 1. PARADIGM 1: Neural Permutation Sorter (Gumbel-Sinkhorn Network)
# =====================================================================
class NeuralPermutationSorter(nn.Module):
    """
    Learns to dynamically construct a permutation matrix P(x) such that:
        y_sorted = P(x) * x
    Uses Sinkhorn-Knopp iterations to project unconstrained scores into
    a doubly stochastic matrix (continuous relaxation of a permutation matrix).
    """
    def __init__(self, n_elements, hidden_dim=64, n_sinkhorn_iters=15, temperature=0.1):
        super().__init__()
        self.n = n_elements
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.temperature = temperature
        
        # Scoring network: maps each element and its context to a ranking score
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def sinkhorn(self, log_alpha):
        """
        Sinkhorn-Knopp algorithm in log-space for numerical stability.
        Normalizes rows and columns alternately to produce a doubly-stochastic matrix.
        """
        for _ in range(self.n_sinkhorn_iters):
            # Row normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            # Column normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha)

    def forward(self, x):
        # x shape: (batch_size, n)
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(-1) # (batch_size, n, 1)
        
        # Compute latent ranking representation for each element
        scores = self.encoder(x_expanded).squeeze(-1) # (batch_size, n)
        
        # Construct pairwise comparison matrix: S_ij = -(scores_i - target_rank_j)^2
        # Or pairwise relative ranks:
        # For rank j in [0, ..., n-1], ideal sorted position corresponds to sorted score
        # NeuralSort formulation (Grover et al., 2019):
        # A_ij = -|scores_i - sorted_score_proxy| or distance to regular grid:
        grid = torch.linspace(0, 1, self.n, device=x.device).unsqueeze(0).repeat(batch_size, 1) # (batch, n)
        
        # Pairwise distance matrix between scores and grid slots
        dist = -torch.abs(scores.unsqueeze(-1) - grid.unsqueeze(-2)) / self.temperature
        
        # Apply Sinkhorn to obtain soft permutation matrix P(x)
        P = self.sinkhorn(dist) # (batch, n, n)
        
        # Matrix multiply: y = P * x
        # P[b, i, j] represents probability that input x[j] belongs to output position i
        y_sorted = torch.bmm(P.transpose(1, 2), x.unsqueeze(-1)).squeeze(-1)
        return y_sorted, P

    def sort_inference(self, arr):
        """Inference with discrete Hungarian/Argmax rounding"""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            _, P = self.forward(x_tensor)
            P_matrix = P[0].cpu().numpy()
            # Match each rank to the most confident input index (argmax assignment)
            assigned_indices = np.argmax(P_matrix, axis=0)
            return arr[assigned_indices]


def train_neural_sorter(n_elements=10, epochs=300, batch_size=64):
    """Trains a NeuralPermutationSorter for a fixed vector size n"""
    model = NeuralPermutationSorter(n_elements=n_elements, hidden_dim=64, temperature=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print(f"[*] Training Neural Permutation Sorter for n={n_elements} over {epochs} epochs...")
    start_t = time.time()
    for epoch in range(epochs):
        # Generate random training batch uniformly in [0, 1]
        x_train = torch.rand(batch_size, n_elements)
        y_target, _ = torch.sort(x_train, dim=-1)
        
        optimizer.zero_grad()
        y_pred, _ = model.forward(x_train)
        loss = criterion(y_pred, y_target)
        loss.backward()
        optimizer.step()
        
    duration = time.time() - start_t
    print(f"[+] Training completed in {duration:.2f}s! Final MSE Loss: {loss.item():.6f}")
    return model


# =====================================================================
# 2. PARADIGM 2: O(n) LearnedSort (CDF Estimation & Bucketing)
# =====================================================================
class LearnedSort:
    """
    O(n) Expected Time Learned Sorter based on CDF estimation.
    Approximates the Cumulative Distribution Function F(x) = P(X <= x).
    Sorted rank index is predicted in O(1) time per element:
        bucket = floor((num_buckets - 1) * F_hat(x))
    Buckets are then sorted locally (Insertion Sort, O(1) for small buckets).
    """
    def __init__(self, sample_ratio=0.05, min_sample=64, max_sample=2048):
        self.sample_ratio = sample_ratio
        self.min_sample = min_sample
        self.max_sample = max_sample

    def _fit_cdf_spline(self, sample):
        """Builds a fast piecewise linear monotonic spline of the empirical CDF"""
        sorted_sample = np.sort(sample)
        m = len(sorted_sample)
        # Quantile thresholds
        quantiles = np.linspace(0.0, 1.0, m)
        return sorted_sample, quantiles

    def sort(self, arr):
        n = len(arr)
        if n <= 32:
            # Base case: local insertion sort
            res = arr.copy()
            for i in range(1, n):
                key = res[i]
                j = i - 1
                while j >= 0 and res[j] > key:
                    res[j + 1] = res[j]
                    j -= 1
                res[j + 1] = key
            return res

        # Step 1: Subsample to estimate data distribution in O(sample_size)
        sample_size = min(self.max_sample, max(self.min_sample, int(n * self.sample_ratio)))
        sample_indices = np.random.choice(n, size=sample_size, replace=False)
        sample = arr[sample_indices]
        sample_sorted, quantiles = self._fit_cdf_spline(sample)

        # Step 2: Determine number of buckets (proportional to n)
        num_buckets = max(4, n // 4)
        buckets = [[] for _ in range(num_buckets)]

        # Step 3: O(n) pass - evaluate CDF and place elements into buckets
        # Using vectorized piecewise linear interpolation
        # np.interp evaluates the CDF F_hat(x) in O(1) amortized per item
        cdf_estimates = np.interp(arr, sample_sorted, quantiles, left=0.0, right=1.0)
        bucket_indices = np.clip((cdf_estimates * (num_buckets - 1)).astype(np.int32), 0, num_buckets - 1)

        for i in range(n):
            buckets[bucket_indices[i]].append(arr[i])

        # Step 4: Local sort each bucket & concatenate
        # If model is accurate, each bucket has ~4 elements -> O(1) per bucket -> O(n) total
        sorted_output = []
        for b in buckets:
            if len(b) > 1:
                # Insertion sort for small buckets
                for i in range(1, len(b)):
                    key = b[i]
                    j = i - 1
                    while j >= 0 and b[j] > key:
                        b[j + 1] = b[j]
                        j -= 1
                    b[j + 1] = key
            sorted_output.extend(b)

        return np.array(sorted_output, dtype=arr.dtype)


# =====================================================================
# 3. PARADIGM 3: Classical Algorithmic Baselines
# =====================================================================
def pure_quicksort(arr):
    """Pure algorithmic recursive QuickSort for baseline comparison"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return pure_quicksort(left) + middle + pure_quicksort(right)

def radix_sort_uint(arr):
    """O(n * k) Radix Sort for integers / discretized floats"""
    # Normalize floats to uint32 for radix sort
    min_v, max_v = np.min(arr), np.max(arr)
    if max_v == min_v:
        return arr.copy()
    scaled = np.uint32((arr - min_v) / (max_v - min_v) * (2**31 - 1))
    
    RADIX = 256
    SHIFT = 8
    MASK = 255
    out = scaled.copy()
    temp = np.empty_like(out)
    
    for byte in range(4):
        counts = np.zeros(RADIX, dtype=np.int32)
        shift_amount = byte * SHIFT
        
        # Counting pass
        for x in out:
            counts[(x >> shift_amount) & MASK] += 1
            
        # Cumulative prefix sum
        prefixes = np.zeros(RADIX, dtype=np.int32)
        prefixes[0] = 0
        for i in range(1, RADIX):
            prefixes[i] = prefixes[i - 1] + counts[i - 1]
            
        # Place elements
        for x in out:
            bucket = (x >> shift_amount) & MASK
            temp[prefixes[bucket]] = x
            prefixes[bucket] += 1
            
        out, temp = temp, out
        
    # Reconstruct sorted order
    sorted_floats = (out.astype(np.float64) / (2**31 - 1)) * (max_v - min_v) + min_v
    return sorted_floats


# =====================================================================
# 4. BENCHMARK & COMPLEXITY TEST RUNNER
# =====================================================================
def verify_sorted(arr):
    """Checks if array is monotonically non-decreasing"""
    return bool(np.all(arr[:-1] <= arr[1:]))

def count_inversions(arr):
    """Counts number of adjacent inversions (order violations)"""
    return int(np.sum(arr[:-1] > arr[1:]))

def run_benchmarks():
    results = {}

    print("=" * 70)
    print("TEST 1: NEURAL PERMUTATION SORTER (Feasibility & Inversion Test)")
    print("=" * 70)
    neural_sizes = [5, 10, 20, 50]
    neural_results = []
    
    for n in neural_sizes:
        model = train_neural_sorter(n_elements=n, epochs=250, batch_size=64)
        
        # Test on 100 unseen test vectors
        test_inversions = []
        test_times = []
        correct_count = 0
        
        for _ in range(100):
            test_arr = np.random.rand(n).astype(np.float32)
            t0 = time.perf_counter()
            pred_sorted = model.sort_inference(test_arr)
            t1 = time.perf_counter()
            test_times.append((t1 - t0) * 1000.0) # ms
            
            invs = count_inversions(pred_sorted)
            test_inversions.append(invs)
            if invs == 0 and len(np.unique(pred_sorted)) == n:
                correct_count += 1
                
        avg_time = np.mean(test_times)
        avg_invs = np.mean(test_inversions)
        acc = (correct_count / 100.0) * 100.0
        
        print(f"[*] n={n:3d} | Avg Latency: {avg_time:6.3f} ms | Avg Inversions: {avg_invs:4.2f} | 100% Exact Sorts: {acc:5.1f}%")
        neural_results.append({
            "n": n,
            "avg_latency_ms": float(avg_time),
            "avg_inversions": float(avg_invs),
            "accuracy_percent": float(acc)
        })
    results["neural_permutation"] = neural_results

    print("\n" + "=" * 70)
    print("TEST 2: O(n) LEARNEDSORT vs CLASSICAL BASELINES (Scaling Test)")
    print("=" * 70)
    
    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    scaling_data = {
        "sizes": sizes,
        "LearnedSort": [],
        "TimSort_python": [],
        "NumPy_Introsort": [],
        "Pure_QuickSort": [],
        "RadixSort": []
    }
    
    learned_sorter = LearnedSort()
    
    for n in sizes:
        print(f"\n[-] Benchmarking array size n = {n} ...")
        # Generate random float array from Gaussian distribution (common real-world distribution)
        arr = np.random.randn(n).astype(np.float64)
        
        # 1. LearnedSort
        t0 = time.perf_counter()
        res_learned = learned_sorter.sort(arr)
        t_learned = (time.perf_counter() - t0) * 1000.0
        assert verify_sorted(res_learned), f"LearnedSort failed verification on n={n}"
        scaling_data["LearnedSort"].append(float(t_learned))
        print(f"    LearnedSort:      {t_learned:8.3f} ms (Exact: {verify_sorted(res_learned)})")

        # 2. Python built-in TimSort (C-accelerated O(n log n))
        py_list = arr.tolist()
        t0 = time.perf_counter()
        res_timsort = sorted(py_list)
        t_timsort = (time.perf_counter() - t0) * 1000.0
        scaling_data["TimSort_python"].append(float(t_timsort))
        print(f"    Python TimSort:   {t_timsort:8.3f} ms")

        # 3. NumPy Introsort (compiled C library)
        t0 = time.perf_counter()
        res_numpy = np.sort(arr)
        t_numpy = (time.perf_counter() - t0) * 1000.0
        scaling_data["NumPy_Introsort"].append(float(t_numpy))
        print(f"    NumPy Introsort:  {t_numpy:8.3f} ms")

        # 4. Pure QuickSort (Python algorithmic) - only up to 10k to prevent stack overflow
        if n <= 10000:
            py_list = arr.tolist()
            t0 = time.perf_counter()
            res_qs = pure_quicksort(py_list)
            t_qs = (time.perf_counter() - t0) * 1000.0
            scaling_data["Pure_QuickSort"].append(float(t_qs))
            print(f"    Pure QuickSort:   {t_qs:8.3f} ms")
        else:
            scaling_data["Pure_QuickSort"].append(None)

        # 5. Radix Sort
        t0 = time.perf_counter()
        res_radix = radix_sort_uint(arr)
        t_radix = (time.perf_counter() - t0) * 1000.0
        scaling_data["RadixSort"].append(float(t_radix))
        print(f"    RadixSort:        {t_radix:8.3f} ms")

    results["scaling_benchmark"] = scaling_data

    # Empirical Complexity Fitting (T(n) = a * n^b => log(T) = b * log(n) + log(a))
    print("\n" + "=" * 70)
    print("EMPIRICAL COMPLEXITY EXPONENT ESTIMATION (T(n) ~ n^b)")
    print("=" * 70)
    exponents = {}
    valid_sizes = np.array(sizes)
    
    for algo in ["LearnedSort", "TimSort_python", "NumPy_Introsort", "RadixSort"]:
        times = np.array(scaling_data[algo])
        # Log-log linear regression
        log_n = np.log(valid_sizes)
        log_t = np.log(times)
        # Slope b is the empirical complexity exponent
        slope, intercept = np.polyfit(log_n, log_t, 1)
        exponents[algo] = float(slope)
        print(f"[*] {algo:<18}: Empirical Exponent b = {slope:.3f} (Theoretical ideal: 1.00 for O(n), ~1.10 for O(n log n))")
        
    results["empirical_exponents"] = exponents

    # Save complete JSON results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[+] Full benchmark results saved to 'benchmark_results.json'!")

if __name__ == "__main__":
    run_benchmarks()
