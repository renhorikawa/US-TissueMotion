import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    if not ret:
        print("動画の読み込みに失敗しました。")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    return cap, cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

def calculate_optical_flow(prvs, next, roi):
    x, y, w, h = roi
    roi_prvs = prvs[y:y+h, x:x+w]
    roi_next = next[y:y+h, x:x+w]

    flow = cv2.calcOpticalFlowFarneback(roi_prvs, roi_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # xとy成分から動きの大きさ（マグニチュード）を計算
    dx = flow[..., 0]  # x方向の動き
    dy = flow[..., 1]  # y方向の動き

    # 光学フローの大きさ（マグニチュード）を計算
    mag = np.sqrt(dx**2 + dy**2)

    return flow, mag

def plot_histogram_of_percentiles(mag, percentiles=np.arange(0, 101, 10), output_file='histogram_output.png'):
    # 動きの大きさ（マグニチュード）の有効なピクセルを取得
    valid_magnitudes = mag[mag > 0]  # 動きがあるピクセルのみ（動きがゼロでない）
    
    if valid_magnitudes.size > 0:
        # パーセンタイルごとの値を計算
        percentile_values = np.percentile(valid_magnitudes, percentiles)

        # ヒストグラムをプロット
        plt.figure(figsize=(10, 6))
        plt.hist(valid_magnitudes, bins=50, color='blue', alpha=0.7, label='Movement Magnitude')
        
        # パーセンタイル位置を赤い破線で表示
        for i in range(len(percentiles)):
            plt.axvline(percentile_values[i], color='r', linestyle='dashed', label=f'{percentiles[i]}th percentile')

        plt.title("Histogram of Movement Magnitudes with Percentiles")
        plt.xlabel("Movement Magnitude")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

        # ヒストグラムをPNGファイルとして保存
        plt.savefig(output_file, format='png')
        plt.close()
        print(f"ヒストグラムが '{output_file}' として保存されました。")
    else:
        print("動きのあるピクセルがありません。")

# 動画を初期化
cap, prvs = initialize_video_capture("assets/echo_data/1cm.mp4")

# 動画の最初のフレームで手動でROIを選択
ret, frame = cap.read()
if not ret:
    print("動画の読み込みに失敗しました。")
    exit()

# cv2.selectROIで手動でROIを選択
roi = cv2.selectROI("ROI選択", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("ROI選択")

# ヒストグラム保存用フォルダを作成
output_folder = 'histograms'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0
histogram_count = 0  # 保存するヒストグラムのカウント

# 動画の各フレームに対して処理を行う
while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 光学フローを計算
    flow, mag_full = calculate_optical_flow(prvs, next, roi)

    # 動きがあるピクセルがあれば、ヒストグラムを保存
    if np.count_nonzero(mag_full) > 0:
        histogram_count += 1
        histogram_file = os.path.join(output_folder, f'histogram_{histogram_count}.png')
        
        # ヒストグラムを作成して保存
        plot_histogram_of_percentiles(mag_full, percentiles=np.arange(0, 101, 10), output_file=histogram_file)

    # ヒストグラムが10個保存されたら終了
    if histogram_count >= 10:
        print("10個のヒストグラムが保存されました。処理を終了します。")
        break

    frame_count += 1
    prvs = next

cap.release()
cv2.destroyAllWindows()
