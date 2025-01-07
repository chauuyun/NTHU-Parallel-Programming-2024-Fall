#include <cstdio>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <mpi.h>
#include<algorithm>
void mergeSort_increasing(float *a, float *neighbor, float *temp, int m, int neighborSize) {
    int temp_idx = 0, a_idx = 0, nei_idx = 0;
    while(temp_idx < m && a_idx < m && nei_idx < neighborSize) {
        if(a[a_idx] < neighbor[nei_idx]) {
            temp[temp_idx++] = a[a_idx++];
        }
        else {
            temp[temp_idx++] = neighbor[nei_idx++];
        }
    }

    while(temp_idx < m && a_idx < m) {
        temp[temp_idx++] = a[a_idx++];
    }

    while(temp_idx < m && nei_idx < neighborSize) {
        temp[temp_idx++] = neighbor[nei_idx++];
    }
    return;
}

void mergeSort_decreasing(float *a, float *neighbor, float *temp, int m, int neighborSize) {
    int temp_idx = m - 1, a_idx = m - 1, nei_idx = neighborSize - 1;
    while((temp_idx >= 0) && (a_idx >= 0) && (nei_idx >= 0)) {
        if(a[a_idx] < neighbor[nei_idx]) {
            temp[temp_idx--] = neighbor[nei_idx--];
        }
        else {
            temp[temp_idx--] = a[a_idx--];
        }
    }

    while(temp_idx >= 0 && a_idx >= 0) {
        temp[temp_idx--] = a[a_idx--];
    }

    while(temp_idx >= 0 && nei_idx >= 0) {
        temp[temp_idx--] = neighbor[nei_idx--];
    }
    return;
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Group WORLD_GROUP, USED_GROUP;
    MPI_Comm USED_COMM = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int temp_idx = atoi(argv[1]);
    
    // 計算每個進程負責的數據量 m = floor(temp_idx / size)
    int m = temp_idx / size;  // 使用整數除法，相當於 floor(temp_idx / size)
    // 左相鄰進程的數據大小
    int lProc = (rank == 0) ? 0 : m;
    int start_idx = m * rank;  // 每個進程負責的數據 start index

    m += (rank == size - 1)? (temp_idx % size) : 0;
    // 判斷左進程與右進程的資料量
    // 判斷條件: 若為倒數第二個 rank，其右進程 m 需要加上 (temp_idx % size)
    int rProc = (temp_idx % size != 0 && rank == size - 2)? m + (temp_idx % size) : m;
    rProc = (rank == size - 1)? 0 : rProc;
    if( temp_idx < size){
        if(rank < temp_idx){
            m = temp_idx / temp_idx;
            start_idx = m * rank;
            rProc = (rank == temp_idx-1)? 0 : m;
            lProc = (rank == 0) ? 0 : m;
        }else{
            m = 0;
            start_idx = 0;
            rProc = 0;
            lProc = 0;
        }
    }
    
    // 動態分配緩衝區
    float* a = new float[m];      // input array
    float* rProcArray = new float[rProc]; // Right proc array
    float* lProcArray = new float[lProc];     // Left proc array
    float* temp = new float[m];   // Merge array: use to swap with a array
    // 打開文件並讀取數據
    MPI_File in_file;
    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file);
    MPI_File_read_at(in_file, sizeof(float) * start_idx, a, m, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&in_file);
    // 進行排序
    boost::sort::spreadsort::float_sort(a, a + m);
    int k = size + 1;
    MPI_Status status;

    bool oddPhase = (rank % 2 == 0) ? 0 : 1; // odd = 1, even = 0
    bool unsorted1, unsorted2;
    bool lProc_exist = (lProc > 0)? 1 : 0;
    bool rProc_exist = (rProc > 0)? 1 : 0;
    
    // 進入奇偶排序迴圈
    while( k -- ){
        bool oddSafe = ((oddPhase) && (m > 0) && (rProc_exist))? 1 : 0;
        bool evenSafe = ((!oddPhase) && (m > 0) && (lProc_exist))? 1 : 0;
        
        if ( oddSafe ){
            MPI_Sendrecv(
                a + m - 1, 1, MPI_FLOAT, rank + 1, 0, 
                rProcArray, 1, MPI_FLOAT, rank + 1, 0, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            unsorted1 = (a[m - 1] > rProcArray[0])? 1 : 0;
        }
        if ( evenSafe ){
            MPI_Sendrecv(
                a, 1, MPI_FLOAT, rank - 1, 0, 
                lProcArray + lProc - 1, 1, MPI_FLOAT, rank - 1, 0, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            unsorted2 = (lProcArray[lProc - 1] > a[0])? 1 : 0;
        }
        // Odd-phase: sort with Right rank process
        if (oddSafe && unsorted1){      
            MPI_Sendrecv(
                a, m, MPI_FLOAT, rank + 1, 0, 
                rProcArray, rProc, MPI_FLOAT, rank + 1, 0, 
                MPI_COMM_WORLD, &status);
            mergeSort_increasing(a, rProcArray, temp, m, rProc);
            std::swap(temp, a);
        }
        // even-phase: sort with Left rank process
        if (evenSafe && unsorted2){
            MPI_Sendrecv(
                a, m, MPI_FLOAT, rank - 1, 0, 
                lProcArray, lProc, MPI_FLOAT, rank - 1, 0, 
                MPI_COMM_WORLD, &status);
            mergeSort_decreasing(a, lProcArray, temp, m, lProc);
            std::swap(temp, a);
        }
        // change odd to even phase or even to odd phase
        oddPhase = (oddPhase == 1)? 0 : 1;
    }
    
    // 寫入排序後的結果到輸出文件
    MPI_File out_file;
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out_file);
    MPI_File_write_at(out_file, sizeof(float) * start_idx, a, m, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&out_file);
    // 釋放資源並結束 MPI
    MPI_Finalize();
}