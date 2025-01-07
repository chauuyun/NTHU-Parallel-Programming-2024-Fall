/*v3_nsys: MPI_Bcast + MPI_Iallreduce + local_buff & global_buff + schedule(static)*/
/*openmpi:
  t10:    0.35
  t1000:  0.41
  t3000:  2.81
  t4000:  6.21
  t5000:  11.80
  t10000: 108.15
*/
/*mpi:
  t10:    0.78
  t1000:  0.68
  t3000:  2.00
  t4000:  3.96
  t5000:  6.98
  t10000: 54.01
*/
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

// Using restrict pointers for hints to the compiler (if supported):
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

    if (rank == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        input(argv[1]);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(DATA, C, MPI_INT, 0, MPI_COMM_WORLD);

    int c = C;
    vector<int> cuts(c+2);
    for (int i = 0; i < c; i++) {
        cuts[i] = DATA[i];
    }
    cuts[c++] = 0;
    cuts[c++] = N;
    sort(cuts.begin(), cuts.end());

    // dp: c*c
    // dp[i][i+1] = 0; else INT_MAX
    vector<int> dp(c*c, INT_MAX);
    for (int i = 0; i < c-1; i++) {
        dp[i*c + (i+1)] = 0;
    }

    // Pre-allocate buffers to avoid repeated allocations
    // Each len iteration has (c-len+1) elements to update
    vector<int> local_buf;
    vector<int> global_buf;

    // Start timing (optional)
    //double start_time = MPI_Wtime();

    for (int len = 3; len <= c; ++len) {
        int total_tasks = c - len + 1;
        if (total_tasks <= 0) continue;

        int chunk_size = total_tasks / size;
        int remainder = total_tasks % size;
        int start_l = rank * chunk_size + (rank < remainder ? rank : remainder);
        int end_l = start_l + chunk_size + (rank < remainder ? 1 : 0);
        if (end_l > total_tasks) end_l = total_tasks;

        local_buf.assign(total_tasks, INT_MAX);

        // Compute partial dp
        // Using static scheduling for better performance
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

        global_buf.resize(total_tasks, INT_MAX);

        // Non-blocking collective reduce
        MPI_Request req;
        MPI_Iallreduce(local_buf.data(), global_buf.data(), total_tasks, MPI_INT, MPI_MIN, MPI_COMM_WORLD, &req);

        // Potentially, here we could do some non-dependent computations or preparations
        // But since the DP strictly depends on updated results, we must wait.
        
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        // Update dp with global results
        for (int idx = 0; idx < total_tasks; ++idx) {
            int l = idx;
            int r = l + len - 1;
            dp[l*c + r] = global_buf[idx];
        }
    }

    int ans = dp[0*c + (c-1)];

    //double end_time = MPI_Wtime();
    //if (rank == 0) {
    //    fprintf(stderr, "Time: %f s\n", end_time - start_time);
    //}

    if (rank == 0) {
        output(argv[2], ans);
    }

    MPI_Finalize();
    return 0;
}
