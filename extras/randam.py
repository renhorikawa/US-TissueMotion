import random

def generate_random_coordinates(width, height, num_points=30):
    # 画像内で重複しない座標を生成
    coordinates = set()  # setを使うことで重複を防ぐ
    while len(coordinates) < num_points:
        x = random.randint(0, width-1)  # 横方向の座標（0 ~ width-1）
        y = random.randint(0, height-1) # 縦方向の座標（0 ~ height-1）
        coordinates.add((x, y))  # 座標をセットに追加（重複を防ぐ）
    
    return list(coordinates)

# 例：画像の幅100、高さ50でランダムな30点の座標を取得
width = 500  # 画像の幅
height = 500  # 画像の高さ
random_coordinates = generate_random_coordinates(width, height)

# 取得した座標を表示
for coord in random_coordinates:
    print(coord)
    