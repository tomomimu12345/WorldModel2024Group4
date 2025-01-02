import numpy as np

# ファイルをロードする
b = np.load("train.npz", allow_pickle=True)

# データを確認する
for key in b:
    print(key)
    print(b[key])  # 各データを出力
