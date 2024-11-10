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
    return cv2.equalizeHist(image)

def calculate_optical_flow(prvs, next, roi):
    x, y, w, h = roi

    roi_prvs = prvs[y:y+h, x:x+w]
    roi_next = next[y:y+h, x:x+w]

    flow = cv2.calcOpticalFlowFarneback(roi_prvs, roi_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    return flow, mag

def calculate_median_movement(mag):
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

total_median_movement = 0.0
frame_count = 0

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # ヒストグラム平坦化
    next = histogram_equalization(next) 

    # 光学フローを計算（1つのROIのみ）
    flow, mag = calculate_optical_flow(prvs, next, roi1)

    # 移動したピクセルの中央値を計算
    median_movement = calculate_median_movement(mag)

    total_median_movement += median_movement
    frame_count += 1

    # 動きの矢印を描画
    draw_arrows(frame2, flow, roi1, mag, (0, 255, 0))

    # ROI領域を描画
    frame2_with_roi = frame2.copy()
    cv2.rectangle(frame2_with_roi, roi1[:2], (roi1[0]+roi1[2], roi1[1]+roi1[3]), (255, 0, 0), 2)

    cv2.imshow('frame2', frame2_with_roi)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESCキーで終了
        break
    prvs = next

# 動きの中央値の合計を表示
print(f"ROIの動きの中央値の合計: {total_median_movement:.2f}")

cap.release()
cv2.destroyAllWindows()
