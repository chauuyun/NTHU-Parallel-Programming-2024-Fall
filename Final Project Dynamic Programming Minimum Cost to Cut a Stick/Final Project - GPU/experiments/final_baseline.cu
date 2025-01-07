#include <iostream>
#include <algorithm>
#include <fstream>
#include <climits>
#include <cuda.h>

#define NT 1024
#define MAX_N 10010
using namespace std;
/* 1547. Minimum Cost to Cut a Stick */
/* Input: N = 7, DATA = [1,3,4,5]
 * Output: 16
 */

// CPU
// N = DATA 範圍
// C = DATA 的數目
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
int minCost(int n, int c, int *cuts);
// GPU
__constant__ int DATA_GPU[MAX_N];
int *dp;
__global__ void min_reduce(int c, int len, int *dp);

int main(int argc, char *argv[])
{
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    input(input_filename);
    int res = minCost(N, C, DATA);
    output(output_filename, res);

    return 0;
}

int minCost(int n, int c, int *cuts)
{
    cuts[c++] = 0;
    cuts[c++] = n;
    sort(cuts, cuts + c);
    int dp_size = c * c;
    int res;

    // Copy DATA to constant memory on GPU
    cudaMemcpyToSymbol(DATA_GPU, cuts, c * sizeof(int));
    // Allocate and initialize dp array on GPU
    cudaMalloc(&dp, dp_size * sizeof(int));
    cudaMemset(dp, 0, dp_size * sizeof(int));
    // Dynamic programming using CUDA
    for (int len = 3; len <= c; ++len)
    {
        int num = c - len + 1;
        int block_num = (num + NT - 1) / NT;
        min_reduce<<<block_num, NT>>>(c, len, dp);
    }
    // Retrieve the result by get dp[0][c-1]
    cudaMemcpy(&res, &dp[0 * c + c - 1], sizeof(int), cudaMemcpyDeviceToHost);
    return res;
}

__global__ void min_reduce(int c, int len, int *dp)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int r = l + len - 1;
    if (r >= c)
        return;
    int mm = INT_MAX;
    for (int k = l + 1; k < r; ++k)
    {
        mm = min(mm, dp[l * c + k] + dp[k * c + r]);
    }
    mm += DATA_GPU[r] - DATA_GPU[l];
    dp[l * c + r] = mm;
}