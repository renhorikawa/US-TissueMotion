import cv2
import numpy as np

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    if not ret:
        print("動画の読み込みに失敗しました。")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    return cap, cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

def histogram_equalization(image):
    # ヒストグラム平坦化
    return cv2.equalizeHist(image)

def equalize_brightness_between_rois(frame, roi1, roi2):
    # ROI1とROI2の輝度平均を揃える処理
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    roi1_area = frame[y1:y1+h1, x1:x1+w1]
    roi2_area = frame[y2:y2+h2, x2:x2+w2]

    mean_roi1 = np.mean(roi1_area)
    mean_roi2 = np.mean(roi2_area)

    # 平均輝度を揃えるためのスケーリング
    if mean_roi1 > mean_roi2:
        scaling_factor = mean_roi1 / mean_roi2
        roi2_area = np.clip(roi2_area * scaling_factor, 0, 255).astype(np.uint8)
    elif mean_roi1 < mean_roi2:
        scaling_factor = mean_roi2 / mean_roi1
        roi1_area = np.clip(roi1_area * scaling_factor, 0, 255).astype(np.uint8)

    # 新しく輝度が揃ったROI領域を元のフレームに戻す
    frame[y1:y1+h1, x1:x1+w1] = roi1_area
    frame[y2:y2+h2, x2:x2+w2] = roi2_area

    return frame

# 動画キャプチャの初期化
cap, frame1_gray = initialize_video_capture("assets/sample3.mp4")

# 最初のフレームを取得
ret, frame1 = cap.read()
if not ret:
    print("動画の読み込みに失敗しました。")
    exit()

# cv2.selectROIで手動でROIを選択
roi1 = cv2.selectROI("ROI選択", frame1, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("ROI選択")

# 2つ目のROIを、最初のROIの直下に設定
x1, y1, w1, h1 = roi1
roi2 = (x1, y1 + h1, w1, h1)  # 2つ目のROIは最初のROIの直下に同じサイズで設定

# 最初のフレームで輝度を等しくする
frame_with_equalized_brightness = equalize_brightness_between_rois(frame1.copy(), roi1, roi2)

# 輝度調整後の画像を保存
cv2.imwrite("equalized_roi_frame.jpg", frame_with_equalized_brightness)  # 輝度調整後のフレームを保存

# ヒストグラム平坦化
frame1_gray_equalized = histogram_equalization(frame1_gray)

# ヒストグラム平坦化後の画像を保存
cv2.imwrite("hist_equalized_frame.jpg", frame1_gray_equalized)  # ヒストグラム平坦化後のフレームを保存

cap.release()
cv2.destroyAllWindows()
