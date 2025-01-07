// #include <stdio.h>
// #include <math.h>
// #include <omp.h>

// int main(int argc, char** argv) {
//     if (argc != 3) {
//         fprintf(stderr, "must provide exactly 2 arguments!\n");
//         return 1;
//     }

//     unsigned long long r = atoll(argv[1]);   // 圓的半徑
//     unsigned long long k = atoll(argv[2]);   // 用於取模的值
// 	unsigned long long beg = ceil(r/sqrt(2));
// 	unsigned long long total_pixels = beg * beg; // 正方形
	
// 	#pragma omp parallel // openmp 的指令: 告訴 compiler 將此區塊的 code 在多個 threads 中平行執行
//     {
//         // int world_size = omp_get_num_threads(); // 獲取 threads 數量
        
// 		int world_size = omp_get_num_threads();
// 		int world_rank = omp_get_thread_num();	 // 獲取當前 thread 的 ID
// 		// 計算每個 process 負責的工作區域
		
// 		unsigned long long start = (world_rank * ((r - beg) / world_size)) + beg; 
// 		unsigned long long end = (world_rank == world_size - 1) ? r : ((world_rank + 1) * ((r - beg) / world_size)) + beg;
// 		unsigned long long local_pixels = 0;
// 		// 每個 process 計算它負責的區域內的 pixel 數
// 		for (unsigned long long x = start; x < end; x++) {
// 			unsigned long long y = ceil(sqrtl(r * r - x * x));
// 			local_pixels += y;
// 			local_pixels %= k;
// 		}
		
// 		#pragma omp critical
// 		{
// 			local_pixels = 2 * local_pixels ;
// 			total_pixels +=  local_pixels;
// 			total_pixels %= k;
// 		}
		
		
// 		// 使用 MPI_Reduce 將所有 process 的像素數進行累加，並將結果傳遞給 rank 0 process 
// 		// MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		
// 		// if (world_rank == 0) {
// 		// 	printf("%llu\n", (4 * total_pixels) % k);
// 		// }
		
// 	}

// 	printf("%llu\n", (4 * total_pixels) % k);
// 	return 0;
// }


// #include <assert.h>
// #include <stdio.h>
// #include <math.h>
// using namespace std;
// int main(int argc, char** argv) {
// 	if (argc != 3) {
// 		fprintf(stderr, "must provide exactly 2 arguments!\n");
// 		return 1;
// 	}
// 	unsigned long long r = atoll(argv[1]);
// 	unsigned long long k = atoll(argv[2]);
// 	unsigned long long pixels = 0;
// 	//cpu_set_t cpuset;
// 	//sched_getaffinity(0, sizeof(cpuset), &cpuset);
// 	//unsigned long long ncpus = CPU_COUNT(&cpuset);

// 	for (unsigned long long x = 0; x < r; x++) {
// 		unsigned long long y = ceil(sqrtl(r*r - x*x));
// 		pixels += y;
// 		pixels %= k;
// 	}
// 	printf("%llu\n", (4 * pixels) % k);
// }
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_THREADS 64

typedef struct {
    unsigned long long r;
    unsigned long long k;
    unsigned long long start;
    unsigned long long end;
    unsigned long long local_pixels;
} ThreadData;

void* calculate_pixels(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    unsigned long long r = data->r;
    unsigned long long k = data->k;
    unsigned long long start = data->start;
    unsigned long long end = data->end;
    unsigned long long local_pixels = 0;

    for (unsigned long long x = start; x < end; x++) {
        unsigned long long y = ceil(sqrtl(r * r - x * x));
        local_pixels += y;
        local_pixels %= k;
    }

    data->local_pixels = local_pixels;
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    unsigned long long r = atoll(argv[1]);   // 圓的半徑
    unsigned long long k = atoll(argv[2]);   // 用於取模的值
    unsigned long long beg = ceil(r / sqrt(2));
    unsigned long long total_pixels = beg * beg; // 正方形內的像素數量

    int num_threads = MAX_THREADS;
    pthread_t threads[MAX_THREADS];
    ThreadData thread_data[MAX_THREADS];

    unsigned long long range = (r - beg) / num_threads;

    // 創建執行緒並分配工作區域
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].r = r;
        thread_data[t].k = k;
        thread_data[t].start = beg + t * range;
        thread_data[t].end = (t == num_threads - 1) ? r : (beg + (t + 1) * range);
        thread_data[t].local_pixels = 0;

        int rc = pthread_create(&threads[t], NULL, calculate_pixels, (void*)&thread_data[t]);
        if (rc) {
            fprintf(stderr, "ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // 等待所有執行緒完成並累加結果
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
        total_pixels += 2 * thread_data[t].local_pixels;
        total_pixels %= k;
    }

    printf("%llu\n", (4 * total_pixels) % k);
    return 0;
}
