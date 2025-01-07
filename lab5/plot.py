import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# 讀取結果
results = []
for filename in os.listdir('results'):
    if filename.endswith('.json'):
        with open(os.path.join('results', filename), 'r') as f:
            data = json.load(f)
        # 從文件名中提取參數
        params = filename.replace('result_', '').replace('.json', '').split('_')
        data['batch_size'] = int(params[0][2:])
        data['seq_len'] = int(params[1][2:])
        data['num_heads'] = int(params[2][2:])
        data['emb_dim'] = int(params[3][2:])
        data['impl'] = params[4]
        data['causal'] = 'causal' in filename
        results.append(data)

# 創建 DataFrame
df = pd.DataFrame(results)

# 繪製圖表
plt.figure(figsize=(10, 6))
for impl in df['impl'].unique():
    subset = df[df['impl'] == impl]
    plt.plot(subset['seq_len'], subset['forward']['time(s)'], label=f'{impl}')
plt.xlabel('序列長度')
plt.ylabel('前向執行時間（秒）')
plt.title('不同實現下前向執行時間與序列長度的關係')
plt.legend()
plt.show()
