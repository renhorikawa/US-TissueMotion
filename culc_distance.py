import math

# 座標 (x1, y1) と (x2, y2)
x1, y1 = 173, 192
x2, y2 = 621, 190

# 実際の距離 (mm)
actual_distance = 31.3

# ピクセル距離の計算
pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 1ピクセルあたりの実際の距離 (mm)
pixel_to_mm = actual_distance / pixel_distance

# 結果を表示
print("ピクセル距離 (pixel_distance):", pixel_distance)
print("1ピクセルあたりの距離:", pixel_to_mm, "mm")


