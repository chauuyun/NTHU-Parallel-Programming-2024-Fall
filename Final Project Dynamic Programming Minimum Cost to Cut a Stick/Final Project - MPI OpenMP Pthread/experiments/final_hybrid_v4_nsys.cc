/*v4: MPI_Iallreduce + MPI_Sendrecv + MPI_Gatherv + schedule(static) */
/*openmpi:
  t10:    0.62
  t1000:  0.66
  t3000:  1.97
  t4000:  3.95
  t5000:  6.80
  t10000: 53.89
*/
/*mpi:
  t10:    0.75
  t1000:  0.68
  t3000:  2.02
  t4000:  4.01
  t5000:  6.97
  t10000: 55.05
*/
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <climits>
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

    // Rank 0 reads the input
    if (rank == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        input(argv[1]);
    }

    // Distribute N, C, and DATA without MPI_Bcast
    if (rank == 0) {
        // Send data to other ranks
        for (int r = 1; r < size; r++) {
            MPI_Send(&N, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            MPI_Send(&C, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            MPI_Send(DATA, C, MPI_INT, r, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&C, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(DATA, C, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

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
        MPI_Request req;
        MPI_Iallreduce(local_buf.data(), global_buf.data(), total_tasks, MPI_INT, MPI_MIN, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        for (int idx = 0; idx < total_tasks; ++idx) {
            int l = idx;
            int r = l + len - 1;
            dp[l*c + r] = global_buf[idx];
        }
    }

    int ans = dp[0*c + (c-1)];

    if (rank == 0) {
        output(argv[2], ans);
    }

    MPI_Finalize();
    return 0;
}
