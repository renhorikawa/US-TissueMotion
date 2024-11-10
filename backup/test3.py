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

def enhance_contrast_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

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

    return np.sum(mag1), np.sum(mag2), flow1, flow2, mag1, mag2, ang1, ang2

def calculate_average_direction(flow, mag):
    # X軸、Y軸のベクトル成分を計算
    total_angle = 0
    total_magnitude = 0
    for y_pos in range(flow.shape[0]):
        for x_pos in range(flow.shape[1]):
            if mag[y_pos, x_pos] > 1:  # 重要な動きのみ
                angle = np.arctan2(flow[y_pos, x_pos, 1], flow[y_pos, x_pos, 0])  # 動きの方向（角度）
                magnitude = mag[y_pos, x_pos]  # 動きの大きさ
                total_angle += angle * magnitude  # 重み付きの方向
                total_magnitude += magnitude  # 重み付きの大きさ
    if total_magnitude > 0:
        average_angle = total_angle / total_magnitude
    else:
        average_angle = 0  # 動きがない場合は方向0度
    return average_angle

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
prvs = enhance_contrast_clahe(prvs)  

# 動画の最初のフレームで手動でROIを選択
ret, frame = cap.read()
if not ret:
    print("動画の読み込みに失敗しました。")
    exit()

# cv2.selectROIで手動でROIを選択
roi1 = cv2.selectROI("ROI選択", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("ROI選択")

# ROIの大きさを計算
roi1_area = roi1[2] * roi1[3]  # 上のROIの面積

# 2つ目のROIを、最初のROIの直下に設定
x1, y1, w1, h1 = roi1
roi2 = (x1, y1 + h1, w1, h1)  # 2つ目のROIは最初のROIの直下に同じサイズで設定
roi2_area = roi2[2] * roi2[3]  # 下のROIの面積

total_motion_magnitude1 = 0.0
total_motion_magnitude2 = 0.0
total_angle1 = 0.0
total_angle2 = 0.0
frame_count = 0

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    next = enhance_contrast_clahe(next) 

    mag1, mag2, flow1, flow2, mag1_full, mag2_full, ang1, ang2 = calculate_optical_flow(prvs, next, roi1, roi2)

    total_motion_magnitude1 += mag1
    total_motion_magnitude2 += mag2
    frame_count += 1

    # 動きの方向の平均を計算
    avg_angle1 = calculate_average_direction(flow1, mag1_full)
    avg_angle2 = calculate_average_direction(flow2, mag2_full)

    total_angle1 += avg_angle1
    total_angle2 += avg_angle2

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

# 総移動量をROI内のピクセル数で割って平均を計算
average_motion1 = total_motion_magnitude1 / roi1_area
average_motion2 = total_motion_magnitude2 / roi2_area

# 動きの大きさと方向の平均を表示
print(f"上のROIの総移動量: {total_motion_magnitude1:.2f}")
print(f"下のROIの総移動量: {total_motion_magnitude2:.2f}")
print(f"上のROIの平均移動量: {average_motion1:.2f}")
print(f"下のROIの平均移動量: {average_motion2:.2f}")
print(f"上のROIの平均方向（度）: {np.degrees(total_angle1 / frame_count):.2f}")
print(f"下のROIの平均方向（度）: {np.degrees(total_angle2 / frame_count):.2f}")

if total_motion_magnitude2 > 0:
    total_motion_ratio = total_motion_magnitude1 / total_motion_magnitude2
else:
    total_motion_ratio = float('inf')

print(f"上の動きと下の動きの比: {total_motion_ratio:.2f}")

cap.release()
cv2.destroyAllWindows()
