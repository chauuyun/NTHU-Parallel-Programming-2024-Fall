#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <algorithm>
// #include <sys/time.h>
// #define US_PER_SEC 1000000
const int INF = ((1 << 30) - 1);
int n, m;
int *Dist; // Dist 為 1D 陣列: Dist[i*n + j]
const int V = 60010;  // Increased to handle up to 60000 vertices
void input(char* inFileName);
void output(char* outFileName);
void block_FW(int B);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int main(int argc, char* argv[]) {
    // struct timeval start, end;
    // double time;
    // gettimeofday(&start, NULL);
    
    input(argv[1]);
    int B = 64;  // Adjust block size as needed: 64
    block_FW(B);
    output(argv[2]);
    
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    Dist = (int*)malloc(n * n * sizeof(int));

    
    // #pragma unroll
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; ++i) {
        Dist[i] =  (i / n == i % n) ? 0 : INF;
    }

    int *hostEdges = (int *)malloc(3 * m * sizeof(int));
    fread(hostEdges, sizeof(int), 3 * m, file);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        Dist[hostEdges[i*3+0] * n + hostEdges[i*3+1]] = hostEdges[i*3+2];
    }
    // fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "wb");

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; ++i) {
        if (Dist[i] >= INF) Dist[i] = INF;
    }
    fwrite(Dist, sizeof(int), n * n, outfile);
    // fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        // printf("Round %d / %d\n", r + 1, round);
        // fflush(stdout);
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        cal(B, r, r, 0, r, 1);
        cal(B, r, r, r + 1, round - r - 1, 1);
        cal(B, r, 0, r, 1, r);
        cal(B, r, r + 1, r, 1, round - r - 1);
        
        /* Phase 3*/
        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, round - r - 1, r);
        cal(B, r, r + 1, 0, r, round - r - 1);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    
    // #pragma omp parallel for schedule(dynamic)

    #pragma omp parallel for schedule(static)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            int k_start = Round * B;
            int k_end = (Round + 1) * B;
            if (k_end > n) k_end = n;

            int block_internal_start_x = b_i * B;
            int block_internal_end_x = (b_i + 1) * B;
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B;

            if (block_internal_end_x > n) block_internal_end_x = n;
            if (block_internal_end_y > n) block_internal_end_y = n;

            for (int k = k_start; k < k_end; ++k) {
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    int ik = i * n + k;
                    __m128i SIMD_ik = _mm_set1_epi32(Dist[ik]);

                    // 使用 SSE 處理 j 的迴圈，每次處理4個元素
                    int j;
                    for (j = block_internal_start_y; j + 7 < block_internal_end_y; j += 8) {
                        // 載入 Dist[i][j..j+3]
                        __m128i SIMD_l = _mm_loadu_si128((__m128i*)&Dist[i*n + j]);
                        // 載入 Dist[k][j..j+3]
                        __m128i SIMD_kj = _mm_loadu_si128((__m128i*)&Dist[k*n + j]);
                        // SIMD_ik + SIMD_kj
                        __m128i SIMD_r = _mm_add_epi32(SIMD_ik, SIMD_kj);
                        // 取 min
                        __m128i SIMD_min = _mm_min_epi32(SIMD_l, SIMD_r);
                        // 將結果寫回
                        _mm_storeu_si128((__m128i*)&Dist[i*n + j], SIMD_min);
                        // 載入 Dist[i][j..j+3]
                        __m128i SIMD_l1 = _mm_loadu_si128((__m128i*)&Dist[i*n + j+4]);
                        // 載入 Dist[k][j..j+3]
                        __m128i SIMD_kj1 = _mm_loadu_si128((__m128i*)&Dist[k*n + j+4]);
                        // SIMD_ik + SIMD_kj
                        __m128i SIMD_r1 = _mm_add_epi32(SIMD_ik, SIMD_kj1);
                        // 取 min
                        __m128i SIMD_min1 = _mm_min_epi32(SIMD_l1, SIMD_r1);
                        // 將結果寫回
                        _mm_storeu_si128((__m128i*)&Dist[i*n + j+4], SIMD_min1);
                    }

                    // 處理不滿4個的殘餘部分
                    #pragma unroll 4      
                    for (; j < block_internal_end_y; j++) {
                        int ij = i*n + j;
                        int kj = k*n + j;
                        Dist[ij] = std::min(Dist[ij], Dist[ik] + Dist[kj]);
                    }
                }
            }
        }
    }
}
