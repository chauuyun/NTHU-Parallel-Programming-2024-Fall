#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <unistd.h>
#include <climits>

using namespace std;

int N, C;
int DATA[100010];
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

/*

dp[i][j] means the minimum cost for segment [i, j]

dp[i][j] = min(k=i+1 .. j-1){ (cost of [i, j]) + dp[i][k] + dp[k][j] }

*/

int minCost(int n, int cnt, int *cuts)
{
    cuts[cnt++] = 0;
    cuts[cnt++] = n;
    sort(cuts, cuts + cnt);
    vector<vector<int>> dp(cnt, vector<int>(cnt));
    for (int len = 3; len <= cnt; ++len)
    {
        for (int l = 0; l + len - 1 < cnt; ++l)
        {
            int r = l + len - 1;
            int mm = INT_MAX;
            for (int k = l + 1; k < r; ++k)
            {
                mm = min(mm, dp[l][k] + dp[k][r]);
            }
            mm += cuts[r] - cuts[l];
            dp[l][r] = mm;
        }
    }
    return dp[0][cnt - 1];
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("usage: ans input_file output_file");
        exit(1);
    }
    input(argv[1]);

    int ans;

    printf("n: %d, cuts: %d\n", N, C);

    ans = minCost(N, C, DATA);

    printf("ans: %d", ans);
    output(argv[2], ans);

    return 0;
}