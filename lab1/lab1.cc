#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    unsigned long long r = atoll(argv[1]);   // 圓的半徑
    unsigned long long k = atoll(argv[2]);   // 用於取模的值
    unsigned long long beg = ceil(r / sqrt(2));
    unsigned long long total_pixels = beg * beg; // 正方形內的像素數量

    // 初始化 MPI 環境
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // 獲取 process 數量

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  // 獲取當前 process 的 rank

    unsigned long long start = (world_rank * ((r - beg) / world_size)) + beg;
    unsigned long long end = (world_rank == world_size - 1) ? r : ((world_rank + 1) * ((r - beg) / world_size)) + beg;
    unsigned long long local_pixels = 0;

    // 每個 process 計算它負責的區域內的 pixel 數
    for (unsigned long long x = start; x < end; x++) {
        unsigned long long y = ceil(sqrtl(r * r - x * x));
        local_pixels += y;
        local_pixels %= k;
    }

    local_pixels = 2 * local_pixels;

    // 使用 MPI_Reduce 將所有 process 的像素數進行累加，並將結果傳遞給 rank 0 process 
    unsigned long long global_pixels;
    MPI_Reduce(&local_pixels, &global_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // rank 0 process: 輸出最終結果
    if (world_rank == 0) {
        global_pixels += total_pixels;
        global_pixels %= k;
        printf("%llu\n", (4 * global_pixels) % k);
    }

    // 結束 MPI 環境
    MPI_Finalize();

    return 0;
}
