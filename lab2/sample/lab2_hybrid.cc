#include <mpi.h>
#include <omp.h>
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
    unsigned long long total_pixels = beg * beg % k;; // 正方形內的像素數量

    // 初始化 MPI 環境
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // 獲取 process 數量

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  // 獲取當前 process 的 rank

    unsigned long long start = (world_rank * ((r - beg) / world_size)) + beg;
    unsigned long long end = (world_rank == world_size - 1) ? r : ((world_rank + 1) * ((r - beg) / world_size)) + beg;
    unsigned long long local_pixels = 0;

    // 使用 OpenMP 進行每個 MPI process 的平行化計算
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_rank = omp_get_thread_num();

        unsigned long long thread_start = start + thread_rank * ((end - start) / num_threads);
        unsigned long long thread_end = (thread_rank == num_threads - 1) ? end : (start + (thread_rank + 1) * ((end - start) / num_threads));
        unsigned long long thread_local_pixels = 0;

        for (unsigned long long x = thread_start; x < thread_end; x++) {
            unsigned long long y = ceil(sqrtl(r * r - x * x));
            thread_local_pixels += y;
            thread_local_pixels %= k;
        }

        #pragma omp critical
		{
			local_pixels += thread_local_pixels;
        	local_pixels %= k;
			
    	}
    }
	local_pixels = 2 * local_pixels % k;

    // 使用 MPI_Reduce 將所有 process 的像素數進行累加，並將結果傳遞給 rank 0 process 
    unsigned long long global_pixels=0;
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