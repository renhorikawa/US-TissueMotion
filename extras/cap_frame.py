import numpy as np
import cv2
import os

cap = cv2.VideoCapture('assets/sample3.mp4')

output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# フレーム目だけを保存するための処理
ret, frame = cap.read()  # 最初のフレームを読み込む
if not ret:
    print("動画の読み込みに失敗しました")
else:
    output_path = os.path.join(output_folder, 'frame_1.png')  # ファイル名をframe_1.pngに固定
    cv2.imwrite(output_path, frame)  # 最初のフレームを保存
    print(f"フレームを保存しました: {output_path}")

cap.release()  # 動画を解放
