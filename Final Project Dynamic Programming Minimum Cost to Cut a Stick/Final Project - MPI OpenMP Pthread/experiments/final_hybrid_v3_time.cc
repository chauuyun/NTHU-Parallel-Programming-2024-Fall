/*v3_nsys: MPI_Bcast + MPI_Iallreduce + local_buff & global_buff + schedule(static)*/
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <climits>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <cstdio>
#include <cstring>

#define MAX_N 100010

#if defined(__GNUC__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

using namespace std;

int N, C;
int DATA[MAX_N];

static inline void input(const char *file)
{
    FILE *f = fopen(file, "rb");
    fread(&N, sizeof(int), 1, f);
    fread(&C, sizeof(int), 1, f);
    for (int i = 0; i < C; ++i)
    {
        fread(&DATA[i], sizeof(int), 1, f);
    }
    fclose(f);
}

static inline void output(const char *file, int ans)
{
    FILE *f = fopen(file, "wb");
    fwrite(&ans, sizeof(int), 1, f);
    fclose(f);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double io_time = 0.0, comp_time = 0.0, comm_time = 0.0;
    double start_t, end_t;

    // I/O: input
    // 只有 rank=0 執行 input，但我們以 MPI_Barrier 確保計時的一致性。
    MPI_Barrier(MPI_COMM_WORLD);
    start_t = MPI_Wtime();
    if (rank == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        input(argv[1]);
    }
    end_t = MPI_Wtime();
    io_time += (end_t - start_t);

    // 通訊: MPI_Bcast
    MPI_Barrier(MPI_COMM_WORLD);
    start_t = MPI_Wtime();
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(DATA, C, MPI_INT, 0, MPI_COMM_WORLD);
    end_t = MPI_Wtime();
    comm_time += (end_t - start_t);

    int c = C;
    vector<int> cuts(c+2);
    for (int i = 0; i < c; i++) {
        cuts[i] = DATA[i];
    }
    cuts[c++] = 0;
    cuts[c++] = N;
    sort(cuts.begin(), cuts.end());

    vector<int> dp(c*c, INT_MAX);
    for (int i = 0; i < c-1; i++) {
        dp[i*c + (i+1)] = 0;
    }

    vector<int> local_buf;
    vector<int> global_buf;

    for (int len = 3; len <= c; ++len) {
        int total_tasks = c - len + 1;
        if (total_tasks <= 0) continue;

        int chunk_size = total_tasks / size;
        int remainder = total_tasks % size;
        int start_l = rank * chunk_size + (rank < remainder ? rank : remainder);
        int end_l = start_l + chunk_size + (rank < remainder ? 1 : 0);
        if (end_l > total_tasks) end_l = total_tasks;

        local_buf.assign(total_tasks, INT_MAX);

        // 計算時間: 計算區段
        MPI_Barrier(MPI_COMM_WORLD);
        start_t = MPI_Wtime();
        #pragma omp parallel for schedule(static)
        for (int idx = start_l; idx < end_l; ++idx) {
            int l = idx;
            int r = l + len - 1;
            int segment_cost = cuts[r] - cuts[l];
            int val = INT_MAX;
            int *dp_ptr = dp.data();
            for (int m = l + 1; m < r; ++m) {
                int left_val = dp_ptr[l*c + m];
                int right_val = dp_ptr[m*c + r];
                if (left_val != INT_MAX && right_val != INT_MAX) {
                    int cost = left_val + right_val + segment_cost;
                    if (cost < val) val = cost;
                }
            }
            local_buf[idx] = val;
        }
        end_t = MPI_Wtime();
        comp_time += (end_t - start_t);

        global_buf.resize(total_tasks, INT_MAX);

        // 通訊時間: MPI_Iallreduce + MPI_Wait
        MPI_Barrier(MPI_COMM_WORLD);
        start_t = MPI_Wtime();
        MPI_Request req;
        MPI_Iallreduce(local_buf.data(), global_buf.data(), total_tasks, MPI_INT, MPI_MIN, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        end_t = MPI_Wtime();
        comm_time += (end_t - start_t);

        for (int idx = 0; idx < total_tasks; ++idx) {
            int l = idx;
            int r = l + len - 1;
            dp[l*c + r] = global_buf[idx];
        }
    }

    int ans = dp[0*c + (c-1)];

    // I/O: output
    MPI_Barrier(MPI_COMM_WORLD);
    start_t = MPI_Wtime();
    if (rank == 0) {
        output(argv[2], ans);
    }
    end_t = MPI_Wtime();
    io_time += (end_t - start_t);

    // 將時間彙整: 可以使用 MPI_Reduce 得到 max 或 avg
    double max_io_time, max_comp_time, max_comm_time;
    MPI_Reduce(&io_time, &max_io_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        fprintf(stderr, "I/O Time: %f s\n", max_io_time);
        fprintf(stderr, "Computation Time: %f s\n", max_comp_time);
        fprintf(stderr, "Communication Time: %f s\n", max_comm_time);
    }

    MPI_Finalize();
    return 0;
}
