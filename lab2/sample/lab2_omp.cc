#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    unsigned long long r = atoll(argv[1]);   // 圓的半徑
    unsigned long long k = atoll(argv[2]);   // 用於取模的值
	unsigned long long beg = ceil(r/sqrt(2));
	unsigned long long total_pixels = beg * beg; // 正方形
	// unsigned long long total_pixels = 0;
	
	
	#pragma omp parallel // openmp 的指令: 告訴 compiler 將此區塊的 code 在多個 threads 中平行執行
    {
        // int world_size = omp_get_num_threads(); // 獲取 threads 數量
        
		int world_size = omp_get_num_threads();
		int world_rank = omp_get_thread_num();	 // 獲取當前 thread 的 ID
		// 計算每個 process 負責的工作區域
		
		unsigned long long start = (world_rank * ((r - beg) / world_size)) + beg; 
		unsigned long long end = (world_rank == world_size - 1) ? r : ((world_rank + 1) * ((r - beg) / world_size)) + beg;
		unsigned long long local_pixels = 0;
		// 每個 process 計算它負責的區域內的 pixel 數
		for (unsigned long long x = start; x < end; x++) {
			unsigned long long y = ceil(sqrtl(r * r - x * x));
			local_pixels += y;
			local_pixels %= k;
		}
		
		#pragma omp critical
		{
			local_pixels = 2 * local_pixels ;
			total_pixels +=  local_pixels;
			total_pixels %= k;
		}
		
		
		
	}

	printf("%llu\n", (4 * total_pixels) % k);
	return 0;
}

