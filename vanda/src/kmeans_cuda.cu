// CUDA K-means implementation compatible with src/common.hpp.
// Synthetic point generation, initialization and final scoring follow CPU exactly.
// GPU performs assignment, centroid sum/count accumulation, reduction and updates.
// Timing reports the iterative GPU computation; host-to-device and device-to-host
// transfers are reported separately for fair interpretation.

#include "common.hpp"
#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kBlockSize = 256;

inline void cuda_check(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

struct DeviceBuffer {
    void *ptr = nullptr;
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) {
        cuda_check(cudaMalloc(&ptr, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct Event {
    cudaEvent_t e = nullptr;
    Event() { cuda_check(cudaEventCreate(&e), "cudaEventCreate"); }
    ~Event() { if (e) cudaEventDestroy(e); }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
};

// Each block performs assignment and accumulates its own centroid statistics
// in shared memory. This avoids global atomics from every data point.
// partials is channel-major: partials[channel * nblocks + block].
// Channel c*(d+1)+d stores the count for cluster c, exactly as a double
// for N <= 2^53 points.
__global__ void assign_partial_kernel(const double *points, const double *centroids,
                                      double *partials, std::uint64_t n,
                                      int k, int d, int nblocks) {
    extern __shared__ double accum[];
    const int channels = k * (d + 1);
    for (int x = threadIdx.x; x < channels; x += blockDim.x) accum[x] = 0.0;
    __syncthreads();

    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const double *p = points + static_cast<std::size_t>(i) * d;
        int best = 0;
        double best_dist = INFINITY;
        for (int c = 0; c < k; ++c) {
            double dist = 0.0;
            for (int j = 0; j < d; ++j) {
                const double diff = p[j] - centroids[c * d + j];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best = c; }
        }
        const int base = best * (d + 1);
        for (int j = 0; j < d; ++j) atomicAdd(&accum[base + j], p[j]);
        atomicAdd(&accum[base + d], 1.0);
    }
    __syncthreads();

    for (int x = threadIdx.x; x < channels; x += blockDim.x) {
        partials[static_cast<std::size_t>(x) * nblocks + blockIdx.x] = accum[x];
    }
}

__global__ void reduce_partials_kernel(const double *partials, double *totals, int nblocks) {
    const int channel = blockIdx.x;
    __shared__ double sh[kBlockSize];
    double sum = 0.0;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        sum += partials[static_cast<std::size_t>(channel) * nblocks + b];
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) totals[channel] = sh[0];
}

__global__ void update_centroids_kernel(double *centroids, const double *totals,
                                        int k, int d) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= k * d) return;
    const int c = x / d;
    const int j = x % d;
    const double count = totals[c * (d + 1) + d];
    if (count > 0.0) centroids[x] = totals[c * (d + 1) + j] / count;
}

inline double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
} // namespace

int main(int argc, char **argv) {
    try {
        const auto cfg = km::parse_args(argc, argv);
        if (cfg.n > (1ULL << 53))
            throw std::invalid_argument("Point count exceeds exact double count range");
        if (cfg.d == INT_MAX)
            throw std::invalid_argument("Dimensions too large");

        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count < 1) throw std::runtime_error("No CUDA device visible; use a GPU PBS allocation");
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp gpu{};
        cuda_check(cudaGetDeviceProperties(&gpu, 0), "cudaGetDeviceProperties");
        if (gpu.major < 6) throw std::runtime_error("GPU requires native double atomicAdd (SM >= 6.0)");

        const std::size_t k = static_cast<std::size_t>(cfg.k);
        const std::size_t d = static_cast<std::size_t>(cfg.d);
        if (cfg.n > static_cast<std::uint64_t>(SIZE_MAX / d))
            throw std::invalid_argument("Points allocation exceeds size_t");
        const std::size_t nd = static_cast<std::size_t>(cfg.n) * d;
        if (nd > SIZE_MAX / sizeof(double))
            throw std::invalid_argument("Points allocation exceeds size_t");
        const std::size_t point_bytes = nd * sizeof(double);
        const std::size_t center_bytes = k * d * sizeof(double);
        const std::size_t channels = k * (d + 1);
        if (channels > static_cast<std::size_t>(INT_MAX))
            throw std::invalid_argument("Too many centroid channels");
        const std::uint64_t nblocks64 = (cfg.n - 1) / kBlockSize + 1;
        if (nblocks64 > static_cast<std::uint64_t>(INT_MAX))
            throw std::invalid_argument("Too many GPU blocks");
        const int nblocks = static_cast<int>(nblocks64);
        if (channels > SIZE_MAX / static_cast<std::size_t>(nblocks) / sizeof(double))
            throw std::invalid_argument("GPU partial accumulation exceeds size_t");
        const std::size_t partial_bytes = channels * static_cast<std::size_t>(nblocks) * sizeof(double);
        const std::size_t total_bytes = channels * sizeof(double);
        if (channels * sizeof(double) > static_cast<std::size_t>(gpu.sharedMemPerBlock))
            throw std::invalid_argument("K*(D+1) exceeds available shared memory per block");

        std::cout << "GPU_DEVICE name=\"" << gpu.name << "\" compute_capability="
                  << gpu.major << '.' << gpu.minor << " memory_bytes=" << gpu.totalGlobalMem << '\n';

        // Same host-side point generation and initial centroids as serial.cpp.
        auto points = km::make_points(cfg, 0, cfg.n);
        auto centroids = km::make_initial_centroids(cfg);

        DeviceBuffer d_points(point_bytes), d_centroids(center_bytes);
        DeviceBuffer d_partials(partial_bytes), d_totals(total_bytes);
        const auto copy_in_start = std::chrono::steady_clock::now();
        cuda_check(cudaMemcpy(d_points.ptr, points.data(), point_bytes, cudaMemcpyHostToDevice), "H2D points");
        cuda_check(cudaMemcpy(d_centroids.ptr, centroids.data(), center_bytes, cudaMemcpyHostToDevice), "H2D centers");
        cuda_check(cudaDeviceSynchronize(), "H2D synchronize");
        const double h2d_s = seconds_since(copy_in_start);

        Event start, stop;
        cuda_check(cudaEventRecord(start.e), "cudaEventRecord(start)");
        for (int iteration = 0; iteration < cfg.iterations; ++iteration) {
            assign_partial_kernel<<<nblocks, kBlockSize, total_bytes>>>(
                static_cast<const double *>(d_points.ptr),
                static_cast<const double *>(d_centroids.ptr),
                static_cast<double *>(d_partials.ptr), cfg.n, cfg.k, cfg.d, nblocks);
            cuda_check(cudaGetLastError(), "assign_partial_kernel launch");

            reduce_partials_kernel<<<static_cast<unsigned>(channels), kBlockSize>>>(
                static_cast<const double *>(d_partials.ptr),
                static_cast<double *>(d_totals.ptr), nblocks);
            cuda_check(cudaGetLastError(), "reduce_partials_kernel launch");

            update_centroids_kernel<<<static_cast<unsigned>((k*d + kBlockSize - 1)/kBlockSize), kBlockSize>>>(
                static_cast<double *>(d_centroids.ptr),
                static_cast<const double *>(d_totals.ptr), cfg.k, cfg.d);
            cuda_check(cudaGetLastError(), "update_centroids_kernel launch");
        }
        cuda_check(cudaEventRecord(stop.e), "cudaEventRecord(stop)");
        cuda_check(cudaEventSynchronize(stop.e), "CUDA iterations synchronize");
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, start.e, stop.e), "cudaEventElapsedTime");
        const double iteration_time_s = static_cast<double>(ms) / 1000.0;

        const auto copy_out_start = std::chrono::steady_clock::now();
        cuda_check(cudaMemcpy(centroids.data(), d_centroids.ptr, center_bytes, cudaMemcpyDeviceToHost),
                   "D2H centers");
        const double d2h_s = seconds_since(copy_out_start);

        const auto scoring_start = std::chrono::steady_clock::now();
        const auto score = km::score_local(points, 0, centroids, cfg.k, cfg.d);
        const double cpu_score_s = seconds_since(scoring_start);

        std::cout << std::setprecision(17)
                  << "GPU_TIMING kernel_loop_s=" << iteration_time_s
                  << " h2d_s=" << h2d_s
                  << " d2h_s=" << d2h_s
                  << " kernel_plus_transfers_s=" << (iteration_time_s + h2d_s + d2h_s)
                  << " cpu_score_s=" << cpu_score_s
                  << " partial_bytes=" << partial_bytes << '\n';
        km::print_result("cuda", cfg, 1, 1, iteration_time_s, 0.0, score, centroids);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "CUDA K-means error: " << e.what() << '\n';
        return 1;
    }
}
