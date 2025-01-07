// Scoreboard: updated {41 246.55} --> {41 236.86}
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
const int INF = ((1 << 30) - 1);
#define BLOCK_SIZE 64
#define threads 512
// CUDA kernels
__global__ void fw_phase1(int *dist, int padded_n, int r);
__global__ void fw_phase2(int *dist, int padded_n, int r, int numBlocks);
__global__ void fw_phase3(int *dist, int padded_n, int r, int numBlocks);
__global__ void setEdges(int *d_mat, int *d_edges, int m, int padded_n);
__global__ void initMatrix(int *d_mat, int padded_n);
__global__ void setEdges(int *d_mat, int *d_edges, int m, int padded_n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        int u = d_edges[3 * idx];
        int v = d_edges[3 * idx + 1];
        int w = d_edges[3 * idx + 2];
        d_mat[u * padded_n + v] = w;
    }
}

__global__ void initMatrix(int *d_mat, int padded_n)
{
    // 一維 Thread 索引
    const unsigned int i = threadIdx.y;
    const unsigned int j = threadIdx.x;
    const int global_i = i + blockIdx.y * 64;
    const int global_j = j + blockIdx.x * 64;

    d_mat[global_i          * padded_n + global_j     ] = global_i      == global_j      ? 0 : INF;
    d_mat[(global_i + 32)   * padded_n + global_j     ] = global_i + 32 == global_j      ? 0 : INF;
    d_mat[global_i          * padded_n + global_j + 32] = global_i      == global_j + 32 ? 0 : INF;
    d_mat[(global_i + 32)   * padded_n + global_j + 32] = global_i + 32 == global_j + 32 ? 0 : INF;
}

/*USE Stream*/
int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    FILE *inFile = fopen(argv[1], "rb");
    int n, m;
    fread(&n, sizeof(int), 1, inFile);
    fread(&m, sizeof(int), 1, inFile);
    int padded_n = ((n + 64 - 1) / 64) * 64;
    size_t sizeMatrix = (size_t)padded_n * padded_n * sizeof(int);

    int *host_matrix = (int *)malloc(sizeMatrix);
    int *hostEdges = (int *)malloc(3 * m * sizeof(int));
    fread(hostEdges, sizeof(int), 3 * m, inFile);
    int *device_matrix = nullptr;
    cudaMalloc(&device_matrix, sizeMatrix);

    initMatrix<<<dim3(padded_n / 64, padded_n / 64), dim3(32, 32)>>>(device_matrix, padded_n);
    
    int *deviceEdges = nullptr;
    size_t sizeEdges = 3 * m * sizeof(int);
    cudaMalloc(&deviceEdges, sizeEdges);

    cudaStream_t stream = 0;
    cudaMemcpyAsync(deviceEdges, hostEdges, sizeEdges, cudaMemcpyHostToDevice, stream);

    {
        int blocks = (m + threads - 1) / threads;
        setEdges<<<blocks, threads>>>(device_matrix, deviceEdges, m, padded_n);
    }

    int numBlocks = padded_n / 64;
    #pragma unroll 
    for (int r = 0; r < numBlocks; r++) {
        fw_phase1<<<dim3(1, 1), dim3(32, 32)>>>(device_matrix, padded_n, r);
        fw_phase2<<<dim3(numBlocks, 2), dim3(32, 32)>>>(device_matrix, padded_n, r, numBlocks);
        fw_phase3<<<dim3(numBlocks, numBlocks), dim3(32, 32)>>>(device_matrix, padded_n, r, numBlocks);
    }
    cudaMemcpyAsync(host_matrix, device_matrix, sizeMatrix, cudaMemcpyDeviceToHost, stream);
    FILE *outFile = fopen(argv[2], "wb");
    #pragma unroll
    for (int i = 0; i < n; i++) {
        fwrite(&host_matrix[i * padded_n], sizeof(int), n, outFile);
    }
    return 0;
}

__global__ void fw_phase1(int *dist, int padded_n, int r)
{
    __shared__ int s[64][64];
    const unsigned int &i = threadIdx.y;
    const unsigned int &j = threadIdx.x;
    int B = r * 64;
    int global_i = i + B;
    int global_j = j + B;
    int I_32 = i + 32, J_32 = j + 32, GLOBAL_I_32 = global_i + 32, GLOBAL_J_32 = global_j + 32, GLOBAL_I_padded = global_i * padded_n;
    s[i][j] = dist[GLOBAL_I_padded + global_j];
    s[I_32][j] = dist[GLOBAL_I_32 * padded_n + global_j];
    s[i][J_32] = dist[GLOBAL_I_padded + GLOBAL_J_32];
    s[I_32][J_32] = dist[GLOBAL_I_32 * padded_n + GLOBAL_J_32];

    __syncthreads();

#pragma unroll 
    for (int k = 0; k < 64; ++k)
    {
        s[i][j] = min(s[i][j], s[i][k] + s[k][j]);
        s[I_32][j] = min(s[I_32][j], s[I_32][k] + s[k][j]);
        s[i][J_32] = min(s[i][J_32], s[i][k] + s[k][J_32]);
        s[I_32][J_32] = min(s[I_32][J_32], s[I_32][k] + s[k][J_32]);
        __syncthreads();
    }
    dist[GLOBAL_I_padded + global_j] = s[i][j];
    dist[GLOBAL_I_32 * padded_n + global_j] = s[I_32][j];
    dist[GLOBAL_I_padded + GLOBAL_J_32] = s[i][J_32];
    dist[GLOBAL_I_32 * padded_n + GLOBAL_J_32] = s[I_32][J_32];
    __syncthreads();
}

__global__ void fw_phase2(int *dist, int padded_n, int r, int numBlocks)
{
    if (blockIdx.x == r) return;
    
    int i = threadIdx.y;
    int j = threadIdx.x;
    int B = r * 64;
    __shared__ int s_row[64][64], s_col[64][64];
    int global_i, global_j;
    if (blockIdx.y == 0) {
        // pivot row
        global_i = r * 64 + i;
        global_j = blockIdx.x * 64 + j;
    } else {
        // pivot column
        global_i = blockIdx.x * 64 + i;
        global_j = r * 64 + j;
    }

    int GLOBAL_I_32 = global_i + 32;
    int GLOBAL_J_32 = global_j + 32;
    int GLOBAL_I_padded = global_i * padded_n;
    int I_32 = i + 32, J_32 = j + 32, B_J_32 = B + J_32, B_I_32 = B + I_32, B_J = B + j, B_I = B + i;
    int idx_1 = dist[GLOBAL_I_padded + global_j];
    int idx_2 = dist[GLOBAL_I_32 * padded_n + global_j];
    int idx_3 = dist[GLOBAL_I_padded + GLOBAL_J_32];
    int idx_4 = dist[GLOBAL_I_32 * padded_n + GLOBAL_J_32];

    s_row[i][j] = dist[GLOBAL_I_padded + B_J];
    s_row[I_32][j] = dist[GLOBAL_I_32 * padded_n + B_J];
    s_row[i][J_32] = dist[GLOBAL_I_padded + B_J_32];
    s_row[I_32][J_32] = dist[GLOBAL_I_32 * padded_n + B_J_32];

    s_col[i][j] = dist[B_I * padded_n + global_j];
    s_col[I_32][j] = dist[B_I_32*padded_n + global_j];
    s_col[i][J_32] = dist[B_I * padded_n + GLOBAL_J_32];
    s_col[I_32][J_32] = dist[B_I_32 * padded_n + GLOBAL_J_32];

    __syncthreads();

#pragma unroll 
    for (int k = 0; k < 64; ++k) {
        idx_1 = min(idx_1, s_row[i][k] + s_col[k][j]);
        idx_2 = min(idx_2, s_row[I_32][k] + s_col[k][j]);
        idx_3 = min(idx_3, s_row[i][k] + s_col[k][J_32]);
        idx_4 = min(idx_4, s_row[I_32][k] + s_col[k][J_32]);
    }

    dist[GLOBAL_I_padded + global_j] = idx_1;
    dist[GLOBAL_I_32 * padded_n + global_j] = idx_2;
    dist[GLOBAL_I_padded + GLOBAL_J_32] = idx_3;
    dist[GLOBAL_I_32 * padded_n + GLOBAL_J_32] = idx_4;
}
__global__ void fw_phase3(int *dist, int padded_n, int r, int numBlocks)
{
    if (blockIdx.x == r || blockIdx.y == r) return;

    __shared__ int s_row[64][64], s_col[64][64]; // s[0]=col, s[1]=row, s[2]=s2

    const unsigned int i = threadIdx.y;
    const unsigned int j = threadIdx.x;
    int B = r * 64;

    int global_i = i + blockIdx.y * 64;
    int global_j = j + B;
    int I_32 = i + 32, J_32 = j + 32;
    int GLOBAL_I_32 = global_i + 32, GLOBAL_J_32 = global_j + 32;
    int GLOBAL_I_padded = global_i * padded_n;

    s_col[i][j] = dist[GLOBAL_I_padded + global_j];
    s_col[I_32][j] = dist[(GLOBAL_I_32)*padded_n + global_j];
    s_col[i][J_32] = dist[GLOBAL_I_padded + GLOBAL_J_32];
    s_col[I_32][J_32] = dist[(GLOBAL_I_32)*padded_n + GLOBAL_J_32];

    global_i = i + B;
    global_j = j + blockIdx.x * 64;
    I_32 = i + 32; J_32 = j + 32; GLOBAL_I_32 = global_i + 32; GLOBAL_J_32 = global_j + 32;
    GLOBAL_I_padded = global_i * padded_n;

    s_row[i][j] = dist[GLOBAL_I_padded + global_j];
    s_row[I_32][j] = dist[GLOBAL_I_32 * padded_n + global_j];
    s_row[i][J_32] = dist[GLOBAL_I_padded + GLOBAL_J_32];
    s_row[I_32][J_32] = dist[GLOBAL_I_32 * padded_n + GLOBAL_J_32];

    __syncthreads();

    global_i = i + blockIdx.y * 64;
    global_j = j + blockIdx.x * 64;
    I_32 = i + 32; J_32 = j + 32; GLOBAL_I_32 = global_i + 32; GLOBAL_J_32 = global_j + 32;
    GLOBAL_I_padded = global_i * padded_n;

    int idx_1 = dist[GLOBAL_I_padded + global_j];
    int idx_2 = dist[GLOBAL_I_32 * padded_n + global_j];
    int idx_3 = dist[GLOBAL_I_padded + GLOBAL_J_32];
    int idx_4 = dist[GLOBAL_I_32 * padded_n + GLOBAL_J_32];
#pragma unroll 
    for (int k = 0; k < 64; ++k)
    {
        idx_1 = min(idx_1, s_col[i][k] + s_row[k][j]);
        idx_2 = min(idx_2, s_col[I_32][k] + s_row[k][j]);
        idx_3 = min(idx_3, s_col[i][k] + s_row[k][J_32]);
        idx_4 = min(idx_4, s_col[I_32][k] + s_row[k][J_32]);
    }

    dist[GLOBAL_I_padded + global_j] = idx_1;
    dist[(GLOBAL_I_32)*padded_n + global_j] = idx_2;
    dist[GLOBAL_I_padded + GLOBAL_J_32] = idx_3;
    dist[(GLOBAL_I_32)*padded_n + GLOBAL_J_32] = idx_4;
}