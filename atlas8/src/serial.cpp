#include "common.hpp"

int main(int argc, char **argv) {
    try {
        const auto cfg = km::parse_args(argc, argv);
        const auto points = km::make_points(cfg, 0, cfg.n);
        auto centroids = km::make_initial_centroids(cfg);
        std::vector<double> sums(static_cast<std::size_t>(cfg.k) * cfg.d);
        std::vector<long long> counts(cfg.k);

        const auto start = std::chrono::steady_clock::now();
        for (int iter = 0; iter < cfg.iterations; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(counts.begin(), counts.end(), 0LL);
            for (std::size_t i = 0; i < points.size() / static_cast<std::size_t>(cfg.d); ++i) {
                const double *point = points.data() + i * cfg.d;
                const int label = km::nearest(point, centroids, cfg.k, cfg.d);
                ++counts[label];
                for (int d = 0; d < cfg.d; ++d)
                    sums[static_cast<std::size_t>(label) * cfg.d + d] += point[d];
            }
            km::update_centroids(centroids, sums, counts, cfg.k, cfg.d);
        }
        const auto stop = std::chrono::steady_clock::now();
        const double time_s = std::chrono::duration<double>(stop - start).count();
        const auto score = km::score_local(points, 0, centroids, cfg.k, cfg.d);
        km::print_result("serial", cfg, 1, 1, time_s, 0.0, score, centroids);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
