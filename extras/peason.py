import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

# SeabornのSet2カラーパレットを設定
sns.set_palette("Set2")

# 1つ目のデータセット（x）
x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

# 2つ目のデータセット（y）を生成
# 各xに対応するyを30個ランダムに生成（平均値を中心に少し散らす）
np.random.seed(42)  # 再現性のための乱数シード
y = np.array([np.random.normal(loc=xi, scale=0.2, size=30) for xi in x])

# xとyのデータを整形して1次元に
x_expanded = np.repeat(x, 30)  # 各xに対応する30個のy値
y_expanded = y.flatten()       # 30個ずつのy値を1次元に変換

# ピアソンの積率相関係数とp値を計算
corr, p_value = pearsonr(x_expanded, y_expanded)

# 結果表示
print(f'ピアソンの積率相関係数: {corr}')
print(f'p値: {p_value}')

# --- 散布図をプロット（透明度とサイズ調整） ---
plt.figure(figsize=(8, 6))
plt.scatter(x_expanded, y_expanded, alpha=0.5, s=20)
plt.plot(x, np.mean(y, axis=1), color='red')
plt.legend()
plt.grid(True)
plt.show()
