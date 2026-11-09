#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace km {

struct Config {
    std::uint64_t n = 100000;
    int k = 8;
    int d = 4;
    int iterations = 20;
    std::uint64_t seed = 1234;
    bool print_centroids = false;
};

inline std::uint64_t parse_u64(const char *s, const char *flag) {
    if (*s == '-' || *s == '\0') throw std::invalid_argument(std::string("Invalid ") + flag);
    std::size_t used = 0;
    const auto result = std::stoull(s, &used, 10);
    if (used != std::string(s).size()) throw std::invalid_argument(std::string("Invalid ") + flag);
    return static_cast<std::uint64_t>(result);
}

inline int parse_int(const char *s, const char *flag) {
    const auto value = parse_u64(s, flag);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string("Too large: ") + flag);
    return static_cast<int>(value);
}

inline Config parse_args(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        if (opt == "--print-centroids") {
            c.print_centroids = true;
            continue;
        }
        if (opt == "--help" || opt == "-h") {
            throw std::invalid_argument("Usage: PROGRAM [--points N] [--clusters K] [--dims D] "
                                        "[--iterations I] [--seed S] [--print-centroids]");
        }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value after " + opt);
        const char *val = argv[++i];
        if (opt == "--points") c.n = parse_u64(val, "--points");
        else if (opt == "--clusters") c.k = parse_int(val, "--clusters");
        else if (opt == "--dims") c.d = parse_int(val, "--dims");
        else if (opt == "--iterations") c.iterations = parse_int(val, "--iterations");
        else if (opt == "--seed") c.seed = parse_u64(val, "--seed");
        else throw std::invalid_argument("Unknown argument: " + opt);
    }
    if (c.k <= 0 || c.d <= 0 || c.iterations <= 0 || c.n < static_cast<std::uint64_t>(c.k))
        throw std::invalid_argument("Require --points >= --clusters >= 1, --dims >= 1, --iterations >= 1");
    if (static_cast<std::uint64_t>(c.k) * static_cast<std::uint64_t>(c.d) >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("K*D exceeds supported MPI count");
    return c;
}

inline std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Deterministic, rank-independent synthetic points. Point g is generated identically
// no matter which process owns it. Clusters are well separated along dimension zero.
inline double coordinate(std::uint64_t g, int dim, const Config &c) {
    const std::uint64_t cluster = g % static_cast<std::uint64_t>(c.k);
    const auto h = mix(c.seed ^ mix(g) ^ mix(static_cast<std::uint64_t>(dim) + 9ULL));
    const double noise = (static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0) - 0.5) * 0.2;
    const double center = (dim == 0) ? (10.0 * static_cast<double>(cluster)) :
        static_cast<double>((cluster * static_cast<std::uint64_t>(dim + 3)) % 7ULL);
    return center + noise;
}

inline std::uint64_t begin_index(std::uint64_t n, int rank, int ranks) {
    const auto q = n / static_cast<std::uint64_t>(ranks);
    const auto rem = n % static_cast<std::uint64_t>(ranks);
    return q * static_cast<std::uint64_t>(rank) +
           std::min(static_cast<std::uint64_t>(rank), rem);
}

inline std::vector<double> make_points(const Config &c, std::uint64_t begin, std::uint64_t end) {
    const std::uint64_t local_n = end - begin;
    if (local_n > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(c.d))
        throw std::invalid_argument("Local point array exceeds size_t");
    std::vector<double> p(static_cast<std::size_t>(local_n) * static_cast<std::size_t>(c.d));
    for (std::uint64_t i = 0; i < local_n; ++i)
        for (int d = 0; d < c.d; ++d)
            p[static_cast<std::size_t>(i) * c.d + d] = coordinate(begin + i, d, c);
    return p;
}

inline std::vector<double> make_initial_centroids(const Config &c) {
    std::vector<double> centers(static_cast<std::size_t>(c.k) * c.d);
    // Initial center k uses global point k, ensuring every version starts identically.
    for (int k = 0; k < c.k; ++k)
        for (int d = 0; d < c.d; ++d)
            centers[static_cast<std::size_t>(k) * c.d + d] = coordinate(k, d, c);
    return centers;
}

inline int nearest(const double *point, const std::vector<double> &centers, int k, int d,
                   double *out_dist2 = nullptr) {
    int best = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (int cluster = 0; cluster < k; ++cluster) {
        double sum = 0.0;
        for (int j = 0; j < d; ++j) {
            const double delta = point[j] - centers[static_cast<std::size_t>(cluster) * d + j];
            sum += delta * delta;
        }
        if (sum < best_dist2) {
            best_dist2 = sum;
            best = cluster;
        }
    }
    if (out_dist2) *out_dist2 = best_dist2;
    return best;
}

inline void update_centroids(std::vector<double> &centroids,
                             const std::vector<double> &sums,
                             const std::vector<long long> &counts, int k, int d) {
    for (int cluster = 0; cluster < k; ++cluster) {
        if (counts[cluster] == 0) continue; // Retain previous center for empty cluster.
        for (int j = 0; j < d; ++j) {
            const auto idx = static_cast<std::size_t>(cluster) * d + j;
            centroids[idx] = sums[idx] / static_cast<double>(counts[cluster]);
        }
    }
}

struct Score {
    double inertia = 0.0; // Sum of squared distances to FINAL centroids.
    std::uint64_t assignment_hash = 0;
    std::uint64_t correct_count = 0; // Synthetic ground-truth cluster label: global index modulo K.
};

inline Score score_local(const std::vector<double> &points, std::uint64_t global_begin,
                         const std::vector<double> &centroids, int k, int d) {
    Score score;
    const std::size_t local_n = points.size() / static_cast<std::size_t>(d);
    for (std::size_t i = 0; i < local_n; ++i) {
        double dist2 = 0.0;
        const int label = nearest(points.data() + i * d, centroids, k, d, &dist2);
        score.inertia += dist2;
        if (label == static_cast<int>((global_begin + i) % static_cast<std::uint64_t>(k)))
            ++score.correct_count;
        // XOR is associative/commutative, so this checksum is invariant to MPI partition.
        score.assignment_hash ^= mix(global_begin + i + 0x5555555555555555ULL) ^
                                 mix(static_cast<std::uint64_t>(label) +
                                     mix(global_begin + i));
    }
    return score;
}

inline void print_result(const char *variant, const Config &c, int ranks, int threads,
                         double time_s, double communication_s,
                         const Score &score, const std::vector<double> &centroids) {
    std::cout << std::setprecision(17);
    std::cout << "RESULT variant=" << variant
              << " points=" << c.n << " clusters=" << c.k << " dims=" << c.d
              << " iterations=" << c.iterations
              << " ranks=" << ranks << " threads=" << threads
              << " time_s=" << time_s << " communication_s=" << communication_s
              << " inertia=" << score.inertia
              << " accuracy=" << (static_cast<double>(score.correct_count) / static_cast<double>(c.n))
              << " assignment_hash=" << score.assignment_hash << '\n';
    if (c.print_centroids) {
        for (int cluster = 0; cluster < c.k; ++cluster) {
            std::cout << "CENTROID " << cluster;
            for (int j = 0; j < c.d; ++j)
                std::cout << ' ' << centroids[static_cast<std::size_t>(cluster) * c.d + j];
            std::cout << '\n';
        }
    }
}

} // namespace km
