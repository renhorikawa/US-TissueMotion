import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 保存先のフォルダを作成
output_folder = 'new_fold'
os.makedirs(output_folder, exist_ok=True)

def process_video_with_histogram_and_percentile(video_path, frame_count=10):
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()

    # 前処理：グレースケール変換とぼかし処理
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)

    # ROIを選択
    x, y, w, h = cv2.selectROI("ROI Selection", old_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("ROI Selection")

    # ROIのマスク作成
    roi_mask = np.zeros(old_gray.shape, dtype=np.uint8)  
    roi_mask = cv2.rectangle(roi_mask, (x, y), (x + w, y + h), (255), -1)

    # 特徴点を検出
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=500, qualityLevel=0.1, minDistance=7, blockSize=7, mask=roi_mask)

    # アフィンオプティカルフローのパラメータ
    lk_params = dict(winSize=(200, 200), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 結果保存用リスト
    movement_histograms = []

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        # 光学フロー計算
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 光学フローのマグニチュードを計算
        dx = good_new[:, 0] - good_old[:, 0]
        dy = good_new[:, 1] - good_old[:, 1]
        mag = np.sqrt(dx**2 + dy**2)

        # 移動量のヒストグラムを作成
        movement_histograms.append(mag)

        # 95パーセンタイルの計算
        top_95_percent_movement = np.percentile(mag, 95)

        # ヒストグラムを表示（デバッグ用）
        plt.figure(figsize=(8, 6))
        plt.hist(mag, bins=50, color='blue', alpha=0.7)
        plt.axvline(top_95_percent_movement, color='red', linestyle='dashed', linewidth=2)
        plt.title(f'Frame {frame_idx + 1} - 95th Percentile: {top_95_percent_movement:.2f} px')
        plt.xlabel('Movement Magnitude (px)')
        plt.ylabel('Frequency')
        plt.grid(True)

        # グラフを保存
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'histogram_frame_{frame_idx + 1}.png'))
        plt.close()

        # 次のフレーム用にデータを更新
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()

    print(f"ヒストグラムと95パーセンタイルを{output_folder}フォルダに保存しました。")

# 動画のパスを指定
video_path = 'assets/echo_data/1_3cm.mp4'
process_video_with_histogram_and_percentile(video_path, frame_count=10)
