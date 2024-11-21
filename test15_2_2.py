import cv2
import numpy as np
import sys

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    if not ret:
        print("動画の読み込みに失敗しました。")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    return cap, cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

def enhance_brightness_and_contrast_with_filter(image, clip_limit=2.0, tile_grid_size=(8, 8), kernel_size=5):
    filtered_image = cv2.blur(image, (kernel_size, kernel_size))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(filtered_image)
    return enhanced_image

def calculate_optical_flow(prvs, next, roi):
    x, y, w, h = roi
    roi_prvs = prvs[y:y+h, x:x+w]
    roi_next = next[y:y+h, x:x+w]
    flow = cv2.calcOpticalFlowFarneback(roi_prvs, roi_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx**2 + dy**2)
    return flow, mag

def draw_arrows(frame, flow, roi, mag, color=(0, 255, 0), scale=10, arrow_length=5, min_magnitude=1):
    x, y, w, h = roi
    for y_pos in range(0, h, scale):
        for x_pos in range(0, w, scale):
            if mag[y_pos, x_pos] > min_magnitude:
                dx, dy = flow[y_pos, x_pos]
                length = min(mag[y_pos, x_pos], arrow_length)
                end_x = int(x + x_pos + length * dx)
                end_y = int(y + y_pos + length * dy)
                thickness = 2
                cv2.arrowedLine(frame, (x + x_pos, y + y_pos), (end_x, end_y), color, thickness, tipLength=0.05)

cap, prvs = initialize_video_capture("assets/echo_data/3_3cm.mp4")

ret, frame = cap.read()
if not ret:
    print("動画の読み込みに失敗しました。")
    exit()

roi = cv2.selectROI("ROI選択", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("ROI選択")

frame_count = 0
total_top_95_percent_movement = 0.0
pixel_to_distance = 0.0698

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    next_gray = enhance_brightness_and_contrast_with_filter(next_gray)

    flow, mag_full = calculate_optical_flow(prvs, next_gray, roi)

    # 上位95%の動きに相当する距離を計算
    valid_magnitudes = mag_full[mag_full > 1]
    if valid_magnitudes.size > 0:
        top_95_percent = np.percentile(valid_magnitudes, 95)
        top_95_percent_movement_px = top_95_percent
        top_95_percent_movement_distance = top_95_percent_movement_px * pixel_to_distance
        total_top_95_percent_movement += top_95_percent_movement_distance
        frame_count += 1
        print(f"フレーム{frame_count}: 上位95%の移動量: {top_95_percent_movement_distance:.2f} cm")

    # 動きが検出されなくても処理を続ける
    draw_arrows(frame2, flow, roi, mag_full, (0, 255, 0))
    frame2_with_roi = frame2.copy()
    cv2.rectangle(frame2_with_roi, roi[:2], (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)
    cv2.imshow('frame2', frame2_with_roi)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    prvs = next_gray

print(f"ROIの上位95%の移動距離の合計: {total_top_95_percent_movement:.2f} mm")

cap.release()
cv2.destroyAllWindows()

# 使用しているライブラリとPythonのバージョンを出力
print(f"使用しているPythonバージョン: {sys.version}")
print("使用しているライブラリ:")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")