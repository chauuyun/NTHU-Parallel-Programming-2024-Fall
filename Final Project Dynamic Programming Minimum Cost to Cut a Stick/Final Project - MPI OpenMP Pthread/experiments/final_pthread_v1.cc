/*Pthread:
  t10:    0.23
  t1000:  0.88
  t3000:  8.85
  t4000:  17.58
  t5000:  29.89
  t10000: 249.79
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <climits>
#include <chrono>
#include <pthread.h>

#define MAX_N 100010
using namespace std;

/* 1547. Minimum Cost to Cut a Stick */
/* Input: n = 7, cuts = [1,3,4,5]
 * Output: 16
 */

// N = DATA 範圍
// C = DATA 的數目
int N, C;
int DATA[MAX_N];

void input(const char *file)
{
    FILE *f = fopen(file, "rb");
    fread(&N, sizeof(int), 1, f);
    fread(&C, sizeof(int), 1, f);
    for (int i = 0; i < C; ++i)
    {
        fread(&DATA[i], sizeof(int), 1, f);
    }
    fclose(f);
}

void output(const char *file, int ans)
{
    FILE *f = fopen(file, "wb");
    fwrite(&ans, sizeof(int), 1, f);
    fclose(f);
}

// 全域變數: dp, cuts
static vector<vector<int>> dp;
static int cuts[MAX_N];
static int c;

// 用於平行化的參數
struct ThreadArgs {
    int len;
    int l_start;
    int l_end;
    int c;
    int *cuts;
    vector<vector<int>> *dp;
};

static void *compute_dp_range(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    int len = args->len;
    int c = args->c;
    int *cuts = args->cuts;
    vector<vector<int>> &dp = *(args->dp);

    for (int l = args->l_start; l < args->l_end; ++l) {
        if (l + len - 1 >= c) break;
        int r = l + len - 1;
        int val = INT_MAX;
        for (int m = l + 1; m < r; ++m) {
            val = min(val, dp[l][m] + dp[m][r]);
        }
        val += cuts[r] - cuts[l];
        dp[l][r] = val;
    }

    pthread_exit(NULL);
}

int minCost(int n, int count, int *original_cuts)
{
    c = count;
    // 附加 0 和 n
    original_cuts[c++] = 0;
    original_cuts[c++] = n;
    sort(original_cuts, original_cuts + c);

    for (int i = 0; i < c; i++) {
        cuts[i] = original_cuts[i];
    }

    dp.assign(c, vector<int>(c, 0));

    // 決定要使用多少 threads
    int T = 2; // 可根據情況調整

    for (int len = 3; len <= c; ++len)
    {
        // 建立 threads
        pthread_t threads[T];
        ThreadArgs args[T];

        int total_tasks = (c - len + 1);
        if (total_tasks < 0) total_tasks = 0;
        int chunk = (total_tasks + T - 1) / T; // 平均分配給 T 個 threads

        for (int i = 0; i < T; i++) {
            int start_idx = i * chunk;
            int end_idx = min(start_idx + chunk, total_tasks);
            args[i].len = len;
            args[i].l_start = start_idx;
            args[i].l_end = end_idx;
            args[i].c = c;
            args[i].cuts = cuts;
            args[i].dp = &dp;

            pthread_create(&threads[i], NULL, compute_dp_range, (void*)&args[i]);
        }

        // 等待所有執行緒完成
        for (int i = 0; i < T; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    return dp[0][c - 1];
}

int main(int argc, char *argv[])
{
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    input(input_filename);
    int ans = minCost(N, C, DATA);
    output(output_filename, ans);

    return 0;
}
