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
int ceilDiv(int a, int b)
{
    return (a + b - 1) / b;
}
int minCost(int n, int c, int *cuts);
// GPU
__constant__ int DATA_GPU[MAX_N];
int *dp, *minData;
__global__ void min_reduce(int c, int len, int *dp);
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
        int block_len_cnt = 64;
        preprocess_min_data<<<dim3(ceilDiv(block_num_l, NT), block_len_cnt), NT>>>(c, len, dp, minData);
        do
        {
            min_reduce<<<dim3(ceilDiv(block_len_cnt, NT), block_num_l), NT>>>(c, block_len_cnt, minData);
            block_len_cnt = ceilDiv(block_len_cnt, NT);
        } while (block_len_cnt > 1);
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

__global__ void min_reduce(int c, int dataLen, int *minData)
{
    __shared__ int smem[NT];
    int l = blockIdx.y;
    int minDataIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    smem[tid] = minDataIdx >= dataLen ? INT_MAX : minData[l * c + minDataIdx];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] = min(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        minData[l * c + blockIdx.x] = smem[0];
    }
}