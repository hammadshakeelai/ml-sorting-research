#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>

// ============================================================================
// 1. LEARNEDSORT: O(n) CDF-Based Machine Learning Sorter in C++
// ============================================================================
class LearnedSort {
public:
    struct SplineModel {
        std::vector<double> knots_x;
        std::vector<double> knots_y;
        int num_knots;

        inline double predict(double x) const {
            if (x <= knots_x.front()) return 0.0;
            if (x >= knots_x.back()) return 1.0;

            // Fast branch: binary search over small knot table (fits in L1 cache)
            auto it = std::lower_bound(knots_x.begin(), knots_x.end(), x);
            int idx = std::max(0, static_cast<int>(it - knots_x.begin()) - 1);
            
            double dx = knots_x[idx + 1] - knots_x[idx];
            if (dx <= 1e-12) return knots_y[idx];
            double t = (x - knots_x[idx]) / dx;
            return knots_y[idx] + t * (knots_y[idx + 1] - knots_y[idx]);
        }
    };

    static SplineModel train_model(const double* arr, size_t n, size_t sample_size = 1024, int num_knots = 64) {
        SplineModel model;
        model.num_knots = num_knots;
        
        // Step 1: Subsample to estimate data distribution in O(1) relative to n
        sample_size = std::min(sample_size, n);
        std::vector<double> sample(sample_size);
        
        // Linear stride sampling for cache efficiency
        size_t stride = std::max<size_t>(1, n / sample_size);
        for (size_t i = 0; i < sample_size; ++i) {
            sample[i] = arr[i * stride];
        }
        std::sort(sample.begin(), sample.end());

        // Step 2: Extract quantiles to build monotonic CDF spline knots
        model.knots_x.resize(num_knots);
        model.knots_y.resize(num_knots);
        for (int k = 0; k < num_knots; ++k) {
            double q = static_cast<double>(k) / (num_knots - 1);
            size_t sample_idx = std::min(sample_size - 1, static_cast<size_t>(q * (sample_size - 1)));
            model.knots_x[k] = sample[sample_idx];
            model.knots_y[k] = q;
        }

        return model;
    }

    static void sort(double* arr, size_t n) {
        if (n <= 32) {
            // Base case insertion sort
            for (size_t i = 1; i < n; ++i) {
                double key = arr[i];
                int64_t j = i - 1;
                while (j >= 0 && arr[j] > key) {
                    arr[j + 1] = arr[j];
                    j--;
                }
                arr[j + 1] = key;
            }
            return;
        }

        // 1. Train lightweight CDF model
        SplineModel model = train_model(arr, n);

        // 2. Determine number of buckets (proportional to n for O(1) elements per bucket)
        size_t num_buckets = std::max<size_t>(8, n / 8);
        std::vector<uint32_t> counts(num_buckets, 0);

        // 3. First O(n) pass: count bucket sizes (histogram)
        std::vector<uint32_t> bucket_ids(n);
        double bucket_scale = static_cast<double>(num_buckets - 1);

        for (size_t i = 0; i < n; ++i) {
            double cdf = model.predict(arr[i]);
            uint32_t b = static_cast<uint32_t>(cdf * bucket_scale);
            if (b >= num_buckets) b = num_buckets - 1;
            bucket_ids[i] = b;
            counts[b]++;
        }

        // 4. Prefix sum offsets for contiguous memory placement
        std::vector<uint32_t> offsets(num_buckets + 1, 0);
        for (size_t b = 0; b < num_buckets; ++b) {
            offsets[b + 1] = offsets[b] + counts[b];
        }

        // 5. Second O(n) pass: scatter elements into contiguous buffer
        std::vector<double> out(n);
        std::vector<uint32_t> current_offsets = offsets;
        for (size_t i = 0; i < n; ++i) {
            uint32_t b = bucket_ids[i];
            out[current_offsets[b]++] = arr[i];
        }

        // 6. Final O(n) pass: local insertion sort per bucket
        for (size_t b = 0; b < num_buckets; ++b) {
            size_t start = offsets[b];
            size_t end = offsets[b + 1];
            size_t len = end - start;

            if (len > 1) {
                if (len <= 48) {
                    // Cache-friendly in-place insertion sort
                    for (size_t i = start + 1; i < end; ++i) {
                        double key = out[i];
                        int64_t j = i - 1;
                        while (j >= static_cast<int64_t>(start) && out[j] > key) {
                            out[j + 1] = out[j];
                            j--;
                        }
                        out[j + 1] = key;
                    }
                } else {
                    // Fallback to std::sort for rare outlier clusters
                    std::sort(out.begin() + start, out.begin() + end);
                }
            }
        }

        // Copy back to source array
        std::memcpy(arr, out.data(), n * sizeof(double));
    }
};

// ============================================================================
// 2. C++ RADIX SORT (LSD Radix Sort for 64-bit IEEE-754 Floats)
// ============================================================================
void float64_radix_sort(double* arr, size_t n) {
    if (n <= 1) return;
    std::vector<uint64_t> u(n);
    std::vector<uint64_t> temp(n);

    // Transform IEEE 754 float into sortable unsigned integer
    for (size_t i = 0; i < n; ++i) {
        uint64_t val;
        std::memcpy(&val, &arr[i], sizeof(double));
        // If sign bit is 1, invert all bits; else flip sign bit
        if (val & 0x8000000000000000ULL) {
            val = ~val;
        } else {
            val |= 0x8000000000000000ULL;
        }
        u[i] = val;
    }

    const int RADIX_BITS = 8;
    const int RADIX_SIZE = 1 << RADIX_BITS;
    const int RADIX_MASK = RADIX_SIZE - 1;

    // 8 passes for 64-bit words
    for (int shift = 0; shift < 64; shift += RADIX_BITS) {
        uint32_t count[RADIX_SIZE] = {0};
        for (size_t i = 0; i < n; ++i) {
            count[(u[i] >> shift) & RADIX_MASK]++;
        }

        uint32_t prefix[RADIX_SIZE];
        prefix[0] = 0;
        for (int i = 1; i < RADIX_SIZE; ++i) {
            prefix[i] = prefix[i - 1] + count[i - 1];
        }

        for (size_t i = 0; i < n; ++i) {
            int bucket = (u[i] >> shift) & RADIX_MASK;
            temp[prefix[bucket]++] = u[i];
        }
        u.swap(temp);
    }

    // Decode back to double
    for (size_t i = 0; i < n; ++i) {
        uint64_t val = u[i];
        if (val & 0x8000000000000000ULL) {
            val &= ~0x8000000000000000ULL;
        } else {
            val = ~val;
        }
        std::memcpy(&arr[i], &val, sizeof(double));
    }
}

// ============================================================================
// 3. BENCHMARK HARNESS
// ============================================================================
bool verify_sorted(const double* arr, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        if (arr[i - 1] > arr[i]) return false;
    }
    return true;
}

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "       HIGH-PERFORMANCE C++ BENCHMARK: LEARNEDSORT vs CLASSICAL ALGORITHMS      " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    std::vector<size_t> test_sizes = {10000, 50000, 100000, 500000, 1000000, 3000000};
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(500.0, 150.0);

    for (size_t n : test_sizes) {
        std::cout << "\n>>> BENCHMARKING ARRAY SIZE N = " << n << " (" << (n * sizeof(double)) / (1024 * 1024) << " MB)" << std::endl;

        // Generate baseline dataset
        std::vector<double> base_data(n);
        for (size_t i = 0; i < n; ++i) {
            base_data[i] = dist(rng);
        }

        // 1. LearnedSort (C++ O(n) Machine Learning Sorter)
        {
            std::vector<double> arr = base_data;
            auto t0 = std::chrono::high_resolution_clock::now();
            LearnedSort::sort(arr.data(), n);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            bool ok = verify_sorted(arr.data(), n);
            std::cout << "  [1] LearnedSort (O(n) ML):  " << std::setw(9) << ms << " ms  | Exact: " << (ok ? "PASS" : "FAIL") << std::endl;
        }

        // 2. std::sort (C++ Introsort - QuickSort + HeapSort + InsertionSort)
        {
            std::vector<double> arr = base_data;
            auto t0 = std::chrono::high_resolution_clock::now();
            std::sort(arr.begin(), arr.end());
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            bool ok = verify_sorted(arr.data(), n);
            std::cout << "  [2] std::sort (Introsort):  " << std::setw(9) << ms << " ms  | Exact: " << (ok ? "PASS" : "FAIL") << std::endl;
        }

        // 3. std::stable_sort (C++ MergeSort)
        {
            std::vector<double> arr = base_data;
            auto t0 = std::chrono::high_resolution_clock::now();
            std::stable_sort(arr.begin(), arr.end());
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            bool ok = verify_sorted(arr.data(), n);
            std::cout << "  [3] std::stable_sort:       " << std::setw(9) << ms << " ms  | Exact: " << (ok ? "PASS" : "FAIL") << std::endl;
        }

        // 4. float64_radix_sort (LSD Radix Sort)
        {
            std::vector<double> arr = base_data;
            auto t0 = std::chrono::high_resolution_clock::now();
            float64_radix_sort(arr.data(), n);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            bool ok = verify_sorted(arr.data(), n);
            std::cout << "  [4] LSD RadixSort (8-pass): " << std::setw(9) << ms << " ms  | Exact: " << (ok ? "PASS" : "FAIL") << std::endl;
        }
    }

    std::cout << "\n[+] Benchmark finished successfully!" << std::endl;
    return 0;
}
