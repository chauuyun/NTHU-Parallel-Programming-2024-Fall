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

/*
coalesce
parallel min reduce
first reduce during load
unroll all
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
int ceilDiv(int a, int b)
{
    return (a + b - 1) / b;
}
int minCost(int n, int c, int *cuts);
// GPU
__constant__ int DATA_GPU[MAX_N];
int *dp, *minData;
__global__ void min_reduce(int c, volatile int *dp);
__global__ void preprocess_min_data(int c, int len, int *dp, int *minData);
__global__ void postprocess_min_data(int c, int len, int *dp, int *minData);

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
    int dp_size = (c + 1) * c;
    int res;

    // Copy DATA to constant memory on GPU
    cudaMemcpyToSymbol(DATA_GPU, cuts, c * sizeof(int));
    // Allocate and initialize dp array on GPU
    cudaMalloc(&dp, dp_size * sizeof(int));
    cudaMemset(dp, 0, dp_size * sizeof(int));
    cudaMalloc(&minData, dp_size * sizeof(int));
    // Dynamic programming using CUDA
    for (int len = 3; len <= c; ++len)
    {
        int block_num_l = c - len + 1;
        preprocess_min_data<<<dim3(ceilDiv(block_num_l, NT), 32), NT>>>(c, len, dp, minData);
        min_reduce<<<dim3(1, block_num_l), 16>>>(c, minData);
        postprocess_min_data<<<block_num_l, 1>>>(c, len, dp, minData);
    }
    // Retrieve the result
    // get dp[c][0]
    cudaMemcpy(&res, &dp[c * c + 0], sizeof(int), cudaMemcpyDeviceToHost);
    return res;
}

__global__ void preprocess_min_data(int c, int len, int *dp, int *minData)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int r = l + len - 1;
    if (r >= c)
        return;
    int leftLen = blockIdx.y + 2;
    int mm = INT_MAX;
    for (; leftLen < len; leftLen += gridDim.y)
    {
        int rightLen = len - leftLen + 1, rightIdx = l + leftLen - 1;
        mm = min(mm, dp[leftLen * c + l] + dp[rightLen * c + rightIdx]);
    }
    minData[l * c + blockIdx.y] = mm;
}

__global__ void postprocess_min_data(int c, int len, int *dp, int *minData)
{
    int l = blockIdx.x;
    int r = l + len - 1;
    int baseCost = DATA_GPU[r] - DATA_GPU[l];
    dp[len * c + l] = baseCost + minData[l * c + 0];
}

__global__ void min_reduce(int c, volatile int *minData)
{
    __shared__ volatile int smem[16];
    int l = blockIdx.y;
    int tid = threadIdx.x;
    smem[tid] = min(minData[l * c + tid], minData[l * c + tid + 16]);
    smem[tid] = min(smem[tid], smem[tid + 8]);
    smem[tid] = min(smem[tid], smem[tid + 4]);
    smem[tid] = min(smem[tid], smem[tid + 2]);
    smem[tid] = min(smem[tid], smem[tid + 1]);
    if (tid == 0)
        minData[l * c + blockIdx.x] = smem[0];
}