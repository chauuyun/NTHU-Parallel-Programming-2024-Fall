#include <stdio.h>

#include <omp.h>

int main(int argc, char** argv) {
    int omp_threads, omp_thread;
    // omp_threads: 總執行 threads 數量
    // omp_thread: 當前 thread ID

#pragma omp parallel // openmp 的指令: 告訴 compiler 將此區塊的 code 在多個 threads 中平行執行
    {
        omp_threads = omp_get_num_threads();
        omp_thread = omp_get_thread_num();
        printf("Hello: thread %2d/%2d\n", omp_thread, omp_threads);
    }
    return 0;
}
