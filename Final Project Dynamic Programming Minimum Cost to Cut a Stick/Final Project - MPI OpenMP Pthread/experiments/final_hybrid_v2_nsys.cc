/*v2*/
/*openmpi:
  t10:    0.61
  t1000:  1.79
  t3000:  24.00
  t4000:  57.88
  t5000:  157.71
  t10000: ***STEP 529600.0 ON nthu-cn01 CANCELLED AT 2024-12-10T14:42:03 DUE TO TIME LIMIT ***
*/
/*mpi:
  t10:    0.64
  t1000:  1.38
  t3000:  19.34
  t4000:  44.72
  t5000:  87.07
  t10000: *** STEP 529605.0 ON nthu-cn01 CANCELLED AT 2024-12-10T14:53:04 DUE TO TIME LIMIT ***
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
using namespace std;

/*
   Hybrid MPI+OpenMP version of Minimum Cost to Cut a Stick (LeetCode #1547)
   Steps:
   1. Read N,C, and cuts from rank=0 and broadcast.
   2. Append 0 and N to cuts, sort them.
   3. Initialize dp as per the solution logic:
      - dp[i][i+1] = 0
      - dp[i][j] = INT_MAX for all others
   4. For len = 3 to c:
       - partition the computation of dp[l][r] among ranks
       - Each rank computes a portion of l
       - Use OpenMP parallel for to speed up within each rank
       - MPI_Allreduce(MPI_MIN) to combine dp results
   5. dp[0][c-1] is the answer.

   Make sure all reduce operations combine correctly.
*/

int N, C;
int DATA[MAX_N];

void input(const char *file)
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

void output(const char *file, int ans)
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

    // Broadcast N, C and DATA
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(DATA, C, MPI_INT, 0, MPI_COMM_WORLD);

    int c = C;
    // Append 0 and N
    vector<int> cuts(c+2);
    for (int i = 0; i < c; i++) {
        cuts[i] = DATA[i];
    }
    cuts[c++] = 0;
    cuts[c++] = N;
    sort(cuts.begin(), cuts.end());

    // Initialize dp
    // dp is c x c
    // dp[i][i+1] = 0, else INT_MAX
    vector<int> dp(c*c, INT_MAX);
    for (int i = 0; i < c-1; i++) {
        dp[i*c + (i+1)] = 0;
    }

    // Main DP computation
    // dp[i][j] = (cuts[j]-cuts[i]) + min_{k in (i+1..j-1)} dp[i][k] + dp[k][j]
    for (int len = 3; len <= c; ++len) {
        int total_tasks = c - len + 1;
        if (total_tasks < 0) total_tasks = 0;

        // Divide tasks among ranks
        int chunk_size = total_tasks / size;
        int remainder = total_tasks % size;
        int start_l = rank * chunk_size + (rank < remainder ? rank : remainder);
        int end_l = start_l + chunk_size + (rank < remainder ? 1 : 0);
        if (end_l > total_tasks) end_l = total_tasks;

        // OpenMP parallel region
        #pragma omp parallel for schedule(dynamic,1)
        for (int idx = start_l; idx < end_l; ++idx) {
            int l = idx;
            int r = l + len - 1;
            if (r >= c) continue;
            int segment_cost = cuts[r] - cuts[l];
            int val = INT_MAX;
            for (int m = l + 1; m < r; ++m) {
                int left_val = dp[l*c + m];
                int right_val = dp[m*c + r];
                if (left_val != INT_MAX && right_val != INT_MAX) {
                    int cost = left_val + right_val + segment_cost;
                    if (cost < val) val = cost;
                }
            }
            dp[l*c + r] = val;
        }

        // Synchronize dp among all ranks
        // Use MPI_Allreduce with MPI_MIN to get the minimal dp values from all ranks
        // But we must be careful: INT_MAX represents no update.
        MPI_Allreduce(MPI_IN_PLACE, dp.data(), c*c, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    int ans = dp[0*c + (c-1)];

    if (rank == 0) {
        output(argv[2], ans);
    }

    MPI_Finalize();
    return 0;
}
