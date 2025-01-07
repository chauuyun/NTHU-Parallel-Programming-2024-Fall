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

    // 初始化 MPI 環境
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // 獲取 process 數量

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  // 獲取當前 process 的 rank: p_id

    unsigned long long local_pixels = 0;

    // 計算每個 process 負責的工作區域
    unsigned long long start = (r / world_size) * world_rank;
    unsigned long long end = (world_rank == world_size - 1) ? r : (r / world_size) * (world_rank + 1);

    // 每個 process 計算它負責的區域內的 pixel 數
    for (unsigned long long x = start; x < end; x++) {
        unsigned long long y = ceil(sqrtl(r * r - x * x));
        local_pixels = local_pixels + y;
        local_pixels = local_pixels % k;
    }

    unsigned long long total_pixels = 0;

    // 使用 MPI_Reduce 將所有 process 的像素數進行累加，並將結果傳遞給 rank 0 process 
    MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // rank 0 process: 輸出最終結果
    if (world_rank == 0) {
        printf("%llu\n", (4 * total_pixels) % k);
    }

    // 結束 MPI 環境
    MPI_Finalize();

    return 0;
}