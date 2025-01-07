#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda.h>

int B, N, d;
float *hostQ, *hostK, *hostV, *hostO;
float *deviceQ, *deviceK, *deviceV, *deviceO;


const int Br = 16;
const int Bc = 16;
const int stream_size = 8;

__global__
void forwardKernel(const float* __restrict__ q, 
                  const float* __restrict__ k, 
                  const float* __restrict__ v, 
                  float* __restrict__ o, 
                  const int seqLength, const int dim, 
                  const int totalColTiles, const float softmaxScale) 
{
    extern __shared__ float sharedMemory[];
    float *shared_Q = sharedMemory;                 
    float *shared_K = shared_Q + (Br * (dim + 1));     
    float *shared_V = shared_K + (Bc * (dim + 1));     
    float *shared_O = shared_V + (Bc * (dim + 1));     
    float *shared_S = shared_O + (Br * (dim + 1));    

    int threadIdxX = threadIdx.x;
    int blockIdxX = blockIdx.x; 
    int rowIdx = blockIdxX * Br + threadIdxX;
    if (rowIdx >= seqLength) return;
    #pragma unroll
    for (int x = 0; x < dim; x++) {
        shared_O[threadIdxX * (dim + 1) + x] = 0.0;
    }

    float maxVal = -INFINITY;
    float sumExp = 0.0;

    #pragma unroll
    for (int x = 0; x < dim; x++) {
        shared_Q[threadIdxX * (dim + 1) + x] = q[rowIdx * dim + x];
    }

    __syncthreads();
    #pragma unroll
    for (int j = 0; j < totalColTiles; j++) {
        int colTileStart = j * Bc; 
        int currentColIdx = colTileStart + threadIdxX;
        float mask = (currentColIdx < seqLength) ? 1.0 : 0.0;
        #pragma unroll
        for (int x = 0; x < dim; x++) {
            shared_K[threadIdxX * (dim + 1) + x] = k[currentColIdx * dim + x] * mask;
            shared_V[threadIdxX * (dim + 1) + x] = v[currentColIdx * dim + x] * mask;
        }

        __syncthreads();
        float localMax = -INFINITY; // m
        #pragma unroll 32
        for (int y = 0; y < Bc; y++) {
            float dotProduct = 0.0;
            #pragma unroll 
            for (int dimIdx = 0; dimIdx < dim; dimIdx++) {
                dotProduct += shared_Q[threadIdxX * (dim + 1) + dimIdx] * shared_K[y * (dim + 1) + dimIdx];
            }
            dotProduct *= softmaxScale;
            shared_S[threadIdxX * Bc + y] = dotProduct;
            localMax = fmaxf(dotProduct, localMax);
        }
        float localSum = 0.0;
        #pragma unroll 32
        for (int y = 0; y < Bc; y++) {
            float val = __expf(shared_S[threadIdxX * Bc + y] - localMax);
            shared_S[threadIdxX * Bc + y] = val;
            localSum += val;
        }
        float newMaxVal = fmaxf(maxVal, localMax);
        float newSumExp = __expf(maxVal - newMaxVal) * sumExp + __expf(localMax - newMaxVal) * localSum;
        #pragma unroll
        for (int dimIdx = 0; dimIdx < dim; dimIdx++) {
            float weightedValue = 0.0;
            #pragma unroll 32
            for (int y = 0; y < Bc; y++) {
                weightedValue += shared_S[threadIdxX * Bc + y] * shared_V[y * (dim + 1) + dimIdx];
            }
            float OiOld = shared_O[threadIdxX * (dim + 1) + dimIdx];
            float factorOld = (sumExp * __expf(maxVal - newMaxVal)) * OiOld;
            float factorNew = __expf(localMax - newMaxVal) * weightedValue;
            shared_O[threadIdxX * (dim + 1) + dimIdx] = (factorOld + factorNew) / newSumExp;
        }
        maxVal = newMaxVal;
        sumExp = newSumExp;

        __syncthreads();
    }
    #pragma unroll
    for (int x = 0; x < dim; x++) {
        o[rowIdx * dim + x] = shared_O[threadIdxX * (dim + 1) + x];
    }
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }
    
    
    FILE *file = fopen(argv[1], "rb");
    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);
    size_t size = B * N * d * sizeof(float);
    int I = N * d;

    cudaStream_t stream[stream_size];
    for(int i = 0; i < stream_size; ++i)
        cudaStreamCreate(&stream[i]);
    cudaMallocAsync((void**)&deviceQ, stream_size * N * d * sizeof(float), stream[0]);
    cudaMallocAsync((void**)&deviceK, stream_size * N * d * sizeof(float), stream[1]);
    cudaMallocAsync((void**)&deviceV, stream_size * N * d * sizeof(float), stream[2]);
    cudaMallocAsync((void**)&deviceO, B * N * d * sizeof(float), stream[3]);
    hostQ = (float *)malloc(stream_size * N * d * sizeof(float));
    hostK = (float *)malloc(stream_size * N * d * sizeof(float));
    hostV = (float *)malloc(stream_size * N * d * sizeof(float));
    hostO = (float *)malloc(size);
    cudaDeviceSynchronize();
    int totalColTiles = (N + Bc - 1) / Bc;
    int totalRowTiles = (N + Br - 1) / Br;
    float softmaxScale = 1 / sqrtf((float)d);
    size_t sharedMemorySize = (Br * (d + 1) + Bc * (d + 1) + Bc * (d + 1) + Br * (d + 1) + Br * Bc) * sizeof(float);
    dim3 blockDim(Br);
    dim3 gridDim(totalRowTiles);

    #pragma unroll
    for (int b = 0; b < B; b++) {
        fread(hostQ + (b % stream_size) * I, sizeof(float), I, file);
        cudaMemcpyAsync(deviceQ + (b % stream_size) * I, hostQ + (b%stream_size) * I, I * sizeof(float), cudaMemcpyHostToDevice, stream[b % stream_size]);
        fread(hostK + (b % stream_size) * I, sizeof(float), I, file);
        cudaMemcpyAsync(deviceK + (b % stream_size) * I, hostK + (b % stream_size) * I, I * sizeof(float), cudaMemcpyHostToDevice, stream[b % stream_size]);
        fread(hostV + (b % stream_size) * I, sizeof(float), I, file);
        cudaMemcpyAsync(deviceV + (b % stream_size) * I, hostV + (b % stream_size) * I, I * sizeof(float), cudaMemcpyHostToDevice, stream[b % stream_size]);
        forwardKernel<<<gridDim, blockDim, sharedMemorySize, stream[b % stream_size]>>>(deviceQ + (b % stream_size) * I, deviceK + (b % stream_size) * I, deviceV + (b % stream_size) * I, deviceO + b * I, N, d, totalColTiles, softmaxScale);
        cudaMemcpyAsync(hostO + b * I, deviceO + b * I, I * sizeof(float), cudaMemcpyDeviceToHost, stream[b % stream_size]);

    }
    cudaDeviceSynchronize();

    // 輸出結果到檔案
    // writeOutput(argv[2]);
    FILE *file2 = fopen(argv[2], "wb");
    fwrite(hostO, sizeof(float), B * N * d, file2);
    
    // cudaFree(deviceQ);
    // cudaFree(deviceK);
    // cudaFree(deviceV);
    // cudaFree(deviceO);
    // free(hostQ);
    // free(hostK);
    // free(hostV);
    // free(hostO);
    return 0;
}
