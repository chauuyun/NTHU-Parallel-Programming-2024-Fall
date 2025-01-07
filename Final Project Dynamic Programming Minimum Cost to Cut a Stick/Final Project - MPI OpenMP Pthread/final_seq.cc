/*
  t10:    0.17
  t1000:  0.28
  t3000:  5.47
  t4000:  16.59
  t5000:  36.76
  t10000: 441.01
  */
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <climits>
#include <chrono>

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

// double sortTime, solveTime;
int minCost(int n, int c, int *cuts)
{
    // auto start = std::chrono::high_resolution_clock::now();

    cuts[c++] = 0;
    cuts[c++] = n;
    sort(cuts, cuts + c);

    // auto end = std::chrono::high_resolution_clock::now();
    // sortTime = std::chrono::duration<double, std::milli>(end - start).count();
    // start = std::chrono::high_resolution_clock::now();

    vector<vector<int>> dp(c, vector<int>(c));
    for (int len = 3; len <= c; ++len)
    {
        for (int l = 0; l + len - 1 < c; ++l)
        {
            int r = l + len - 1;
            int val = INT_MAX;
            for (int m = l + 1; m < r; ++m)
            {
                val = min(val, dp[l][m] + dp[m][r]);
            }
            val += cuts[r] - cuts[l];
            dp[l][r] = val;
        }
    }

    // end = std::chrono::high_resolution_clock::now();
    // solveTime = std::chrono::duration<double, std::milli>(end - start).count();
    return dp[0][c - 1];
}

int main(int argc, char *argv[])
{
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    input(input_filename);
    int ans = minCost(N, C, DATA);
    output(output_filename, ans);

    // printf("sort time: %.3lf ms\n", sortTime);
    // printf("solve time: %.3lf ms\n", solveTime);

    return 0;
}