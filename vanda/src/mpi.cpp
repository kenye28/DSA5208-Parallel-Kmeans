#include "common.hpp"
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        const auto cfg = km::parse_args(argc, argv);
        const std::uint64_t begin = km::begin_index(cfg.n, rank, ranks);
        const std::uint64_t end = km::begin_index(cfg.n, rank + 1, ranks);
        const auto points = km::make_points(cfg, begin, end);
        auto centroids = km::make_initial_centroids(cfg);
        const int kd = cfg.k * cfg.d;
        std::vector<double> local_sums(kd), global_sums(kd);
        std::vector<long long> local_counts(cfg.k), global_counts(cfg.k);

        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        double comm_s = 0.0;
        for (int iter = 0; iter < cfg.iterations; ++iter) {
            std::fill(local_sums.begin(), local_sums.end(), 0.0);
            std::fill(local_counts.begin(), local_counts.end(), 0LL);
            for (std::size_t i = 0; i < points.size() / static_cast<std::size_t>(cfg.d); ++i) {
                const double *point = points.data() + i * cfg.d;
                const int label = km::nearest(point, centroids, cfg.k, cfg.d);
                ++local_counts[label];
                for (int d = 0; d < cfg.d; ++d)
                    local_sums[static_cast<std::size_t>(label) * cfg.d + d] += point[d];
            }
            const double c0 = MPI_Wtime();
            MPI_Allreduce(local_sums.data(), global_sums.data(), kd, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            MPI_Allreduce(local_counts.data(), global_counts.data(), cfg.k, MPI_LONG_LONG_INT,
                          MPI_SUM, MPI_COMM_WORLD);
            comm_s += MPI_Wtime() - c0;
            km::update_centroids(centroids, global_sums, global_counts, cfg.k, cfg.d);
        }
        const double local_time = MPI_Wtime() - start;
        double max_time = 0.0, max_comm = 0.0;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm_s, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        const auto local_score = km::score_local(points, begin, centroids, cfg.k, cfg.d);
        km::Score score;
        const auto local_hash = static_cast<unsigned long long>(local_score.assignment_hash);
        unsigned long long global_hash = 0;
        MPI_Reduce(&local_score.inertia, &score.inertia, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0,
                   MPI_COMM_WORLD);
        score.assignment_hash = static_cast<std::uint64_t>(global_hash);
        const auto local_correct = static_cast<unsigned long long>(local_score.correct_count);
        unsigned long long global_correct = 0;
        MPI_Reduce(&local_correct, &global_correct, 1, MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        score.correct_count = static_cast<std::uint64_t>(global_correct);
        if (rank == 0) km::print_result("mpi", cfg, ranks, 1, max_time, max_comm,
                                         score, centroids);
    } catch (const std::exception &e) {
        std::cerr << "Rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
