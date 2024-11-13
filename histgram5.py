import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Seabornのスタイル設定（"Set2"パレットを使用してパステル調に）
sns.set(style="whitegrid")  # グリッドスタイル
sns.set_palette("Set2")  # "Set2"パレットを使用

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

def plot_histograms(mag, frame, roi, output_file='histograms.png'):
    # 動きの大きさ（マグニチュード）の有効なピクセルを取得
    valid_magnitudes = mag[mag > 0]  # 動きがあるピクセルのみ（動きがゼロでない）

    # ROI内の輝度値を取得（グレースケール画像）
    x, y, w, h = roi
    roi_gray = frame[y:y+h, x:x+w]

    # ROI内の輝度値のヒストグラムを計算（ビン数を50に変更）
    roi_hist = cv2.calcHist([roi_gray], [0], None, [50], [0, 256])  # ビン数を50に変更

    # 動きの95%タイルを計算
    percentile_95 = np.percentile(valid_magnitudes, 95)

    # ヒストグラムを別々のグラフで表示
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # グラフのサイズを調整

    # 1つ目のグラフ：光学フローのヒストグラム
    axs[0].hist(valid_magnitudes, bins=50, color=sns.color_palette()[0], alpha=0.7, label='Movement Magnitude')
    axs[0].axvline(percentile_95, color='red', linestyle='dashed', label=f'95th Percentile: {percentile_95:.2f}')
    axs[0].set_title("Histogram of Movement Magnitudes", fontsize=16, weight='bold')  # タイトルのフォントサイズ
    axs[0].set_xlabel("Movement Magnitude", fontsize=14)
    axs[0].set_ylabel("Frequency", fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].grid(True)

    # 2つ目のグラフ：ROI内の輝度値のヒストグラム（棒グラフ）
    axs[1].bar(np.arange(50), roi_hist.flatten(), color=sns.color_palette()[1], alpha=0.7, label='ROI Luminance Histogram')
    axs[1].set_title("Histogram of ROI Luminance", fontsize=16, weight='bold')  # タイトルのフォントサイズ
    axs[1].set_xlabel("Pixel Intensity", fontsize=14)
    axs[1].set_ylabel("Frequency", fontsize=14)
    axs[1].legend(fontsize=12)
    axs[1].grid(True)

    # ヒストグラムをPNGファイルとして保存
    plt.tight_layout()  # グラフのレイアウトを整える
    plt.savefig(output_file, format='png')
    plt.close()
    print(f"ヒストグラムが '{output_file}' として保存されました。")

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

# ヒストグラム保存用フォルダを作成（histgramsフォルダ）
output_folder = 'histgrams'
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
        plot_histograms(mag_full, frame2, roi, output_file=histogram_file)

    # ヒストグラムが10個保存されたら終了
    if histogram_count >= 10:
        print("10個のヒストグラムが保存されました。処理を終了します。")
        break

    frame_count += 1
    prvs = next

cap.release()
cv2.destroyAllWindows()

