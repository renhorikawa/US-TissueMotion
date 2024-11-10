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

def calculate_optical_flow(prvs, next, roi):
    x, y, w, h = roi
    roi_prvs = prvs[y:y+h, x:x+w]
    roi_next = next[y:y+h, x:x+w]

    flow = cv2.calcOpticalFlowFarneback(roi_prvs, roi_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 光学フローのdx, dyを取得
    dx = flow[..., 0]
    dy = flow[..., 1]

    # atan2を使って角度を計算（ラジアン単位）
    ang_rad = np.arctan2(dy, dx)

    # ラジアンを度に変換
    ang_deg = np.degrees(ang_rad)

    # 光学フローの大きさ（マグニチュード）も必要
    mag = np.sqrt(dx**2 + dy**2)

    return flow, mag, ang_deg

def calculate_median_movement(mag):
    # 移動したピクセルのマグニチュードの中央値を計算
    valid_magnitudes = mag[mag > 1]  # 動きがあるピクセルのみ
    if valid_magnitudes.size > 0:
        return np.median(valid_magnitudes)
    else:
        return 0  # 動きがない場合は0

def calculate_median_angle(ang_deg):
    # 動きの角度の中央値を計算（度数法）
    valid_angles = ang_deg[ang_deg > 0]  # 動きがあるピクセルのみ
    if valid_angles.size > 0:
        return np.median(valid_angles)  # すでに度数法なのでそのまま
    else:
        return 0  # 動きがない場合は0

def draw_arrows(frame, flow, roi, mag, ang, color):
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
roi = cv2.selectROI("ROI選択", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("ROI選択")

total_median_movement = 0.0  # 動きの中央値の合計
total_median_angle = 0.0  # 角度の中央値の合計
frame_count = 0

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical Flowの計算
    flow, mag_full, ang_full = calculate_optical_flow(prvs, next, roi)

    # 移動したピクセルの中央値を計算
    median_movement = calculate_median_movement(mag_full)
    # 角度の中央値を計算
    median_angle = calculate_median_angle(ang_full)

    # 動きの中央値は合計
    total_median_movement += median_movement
    # 角度の中央値は合計
    total_median_angle += median_angle
    frame_count += 1

    # 動きの矢印を描画
    draw_arrows(frame2, flow, roi, mag_full, ang_full, (0, 255, 0))

    # ROI領域を描画
    frame2_with_roi = frame2.copy()
    cv2.rectangle(frame2_with_roi, roi[:2], (roi[0]+roi[2], roi[1]+roi[3]), (255, 0, 0), 2)

    cv2.imshow('frame2', frame2_with_roi)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESCキーで終了
        break
    prvs = next

# 動きの中央値の合計を出力
print(f"ROIの動きの中央値の合計: {total_median_movement:.2f}")

# 角度の中央値の平均を計算
average_angle = total_median_angle / frame_count

# 角度の平均を出力
print(f"ROIの動きの角度の中央値の平均: {average_angle:.2f}度")

cap.release()
cv2.destroyAllWindows()
