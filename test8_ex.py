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

    # xとy成分から動きの大きさ（マグニチュード）を計算
    dx = flow[..., 0]  # x方向の動き
    dy = flow[..., 1]  # y方向の動き

    # 光学フローの大きさ（マグニチュード）を計算
    mag = np.sqrt(dx**2 + dy**2)

    return flow, mag

def draw_arrows(frame, flow, roi, mag, color=(0, 255, 0), scale=10, arrow_length=5, min_magnitude=1):
    x, y, w, h = roi
    for y_pos in range(0, h, scale):  # 間引き処理で矢印間隔を大きくする
        for x_pos in range(0, w, scale):
            if mag[y_pos, x_pos] > min_magnitude:  # 動きが小さい部分には描画しない
                # 矢印の方向（角度）を取得
                dx, dy = flow[y_pos, x_pos]
                length = min(mag[y_pos, x_pos], arrow_length)  # 矢印の長さを制限
                end_x = int(x + x_pos + length * dx)
                end_y = int(y + y_pos + length * dy)

                # 矢印の太さを固定（動的に変更せず、一定にする）
                thickness = 2

                # 矢印を描画
                cv2.arrowedLine(frame, (x + x_pos, y + y_pos), (end_x, end_y), color, thickness, tipLength=0.05)

def calculate_median_movement(mag):
    # 移動したピクセルのマグニチュードの中央値を計算
    valid_magnitudes = mag[mag > 1]  # 動きがあるピクセルのみ
    if valid_magnitudes.size > 0:
        return np.median(valid_magnitudes)  # 中央値を計算
    else:
        return 0  # 動きがない場合は0

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

frame_count = 0

# 合計の動きの中央値を保存する変数
total_median_movement = 0.0

# 動画の各フレームに対して処理を行う
while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 光学フローを計算
    flow, mag_full = calculate_optical_flow(prvs, next, roi)

    # フレームごとに動きの中央値を計算
    median_movement = calculate_median_movement(mag_full)
    total_median_movement += median_movement  # 中央値の合計を更新

    frame_count += 1

    # フレームごとの動きを出力
    print(f"フレーム{frame_count}: 動きの中央値: {median_movement:.2f}")

    # 動きの矢印を描画
    draw_arrows(frame2, flow, roi, mag_full, (0, 255, 0))

    # ROI領域を描画
    frame2_with_roi = frame2.copy()
    cv2.rectangle(frame2_with_roi, roi[:2], (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)

    # フレームを表示
    cv2.imshow('frame2', frame2_with_roi)

    # ESCキーで終了
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    prvs = next

# 最終的な動きの中央値の合計を出力
print(f"ROIの動きの中央値の合計: {total_median_movement:.2f}")

cap.release()
cv2.destroy
