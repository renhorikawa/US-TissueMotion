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

def calculate_optical_flow(prvs, next, roi1, roi2):
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    roi_prvs1 = prvs[y1:y1+h1, x1:x1+w1]
    roi_next1 = next[y1:y1+h1, x1:x1+w1]
    
    roi_prvs2 = prvs[y2:y2+h2, x2:x2+w2]
    roi_next2 = next[y2:y2+h2, x2:x2+w2]

    flow1 = cv2.calcOpticalFlowFarneback(roi_prvs1, roi_next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow2 = cv2.calcOpticalFlowFarneback(roi_prvs2, roi_next2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag1, ang1 = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
    mag2, ang2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])

    return flow1, flow2, mag1, mag2

def calculate_median_movement(mag):
    # 移動したピクセルのマグニチュードの中央値を計算
    valid_magnitudes = mag[mag > 1]  # 動きがあるピクセルのみ
    if valid_magnitudes.size > 0:
        return np.median(valid_magnitudes)
    else:
        return 0  # 動きがない場合は0

def draw_arrows(frame, flow, roi, mag, color):
    x, y, w, h = roi
    for y_pos in range(0, h, 5):
        for x_pos in range(0, w, 5):
            if mag[y_pos, x_pos] > 1:
                cv2.arrowedLine(frame,
                                (x + x_pos, y + y_pos),
                                (x + x_pos + int(flow[y_pos, x_pos, 0]), 
                                 y + y_pos + int(flow[y_pos, x_pos, 1])),
                                color, 1, tipLength=3)

cap, prvs = initialize_video_capture("assets/sample3.mp4")

# 動画の最初のフレームで手動でROIを選択
ret, frame = cap.read()
if not ret:
    print("動画の読み込みに失敗しました。")
    exit()

# cv2.selectROIで手動でROIを選択
roi1 = cv2.selectROI("ROI選択", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("ROI選択")

# 2つ目のROIを、最初のROIの直下に設定
x1, y1, w1, h1 = roi1
roi2 = (x1, y1 + h1, w1, h1)  # 2つ目のROIは最初のROIの直下に同じサイズで設定

total_median_movement1 = 0.0
total_median_movement2 = 0.0
frame_count = 0

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 2つのROIの輝度を等しくする
    frame2 = equalize_brightness_between_rois(frame2, roi1, roi2)

    # ヒストグラム平坦化
    next = histogram_equalization(next) 

    flow1, flow2, mag1_full, mag2_full = calculate_optical_flow(prvs, next, roi1, roi2)

    # 移動したピクセルの中央値を計算
    median_movement1 = calculate_median_movement(mag1_full)
    median_movement2 = calculate_median_movement(mag2_full)

    total_median_movement1 += median_movement1
    total_median_movement2 += median_movement2
    frame_count += 1

    # 動きの矢印を描画
    draw_arrows(frame2, flow1, roi1, mag1_full, (0, 255, 0))
    draw_arrows(frame2, flow2, roi2, mag2_full, (255, 0, 0))

    # ROI領域を描画
    frame2_with_roi = frame2.copy()
    cv2.rectangle(frame2_with_roi, roi1[:2], (roi1[0]+roi1[2], roi1[1]+roi1[3]), (255, 0, 0), 2)
    cv2.rectangle(frame2_with_roi, roi2[:2], (roi2[0]+roi2[2], roi2[1]+roi2[3]), (0, 255, 0), 2)

    cv2.imshow('frame2', frame2_with_roi)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESCキーで終了
        break
    prvs = next

# 動きの中央値を表示
print(f"上のROIの動きの中央値: {total_median_movement1 / frame_count:.2f}")
print(f"下のROIの動きの中央値: {total_median_movement2 / frame_count:.2f}")

if total_median_movement2 > 0:
    total_movement_ratio = total_median_movement1 / total_median_movement2
else:
    total_movement_ratio = float('inf')

print(f"上の動きと下の動きの比: {total_movement_ratio:.2f}")

cap.release()
cv2.destroyAllWindows()
