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

    mag1, _ = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
    mag2, _ = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])

    return np.sum(mag1), np.sum(mag2), flow1, flow2, mag1, mag2

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

# ROIの定義 任意の座標を入力する
roi1 = (406, 100, 100, 30)
roi2 = (406, 130, 100, 30)
total_motion_magnitude1 = 0.0
total_motion_magnitude2 = 0.0
frame_count = 0

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    next = enhance_contrast_clahe(next) 

    mag1, mag2, flow1, flow2, mag1_full, mag2_full = calculate_optical_flow(prvs, next, roi1, roi2)

    total_motion_magnitude1 += mag1
    total_motion_magnitude2 += mag2
    frame_count += 1

    draw_arrows(frame2, flow1, roi1, mag1_full, (0, 255, 0))
    draw_arrows(frame2, flow2, roi2, mag2_full, (255, 0, 0))

    frame2_with_roi = frame2.copy()
    cv2.rectangle(frame2_with_roi, roi1[:2], (roi1[0]+roi1[2], roi1[1]+roi1[3]), (255, 0, 0), 2)
    cv2.rectangle(frame2_with_roi, roi2[:2], (roi2[0]+roi2[2], roi2[1]+roi2[3]), (0, 255, 0), 2)

    cv2.imshow('frame2', frame2_with_roi)

    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
    prvs = next

print(f"上のROIの動きの大きさ: {total_motion_magnitude1:.2f}")
print(f"下のROIの動きの大きさ: {total_motion_magnitude2:.2f}")

if total_motion_magnitude2 > 0:
    total_motion_ratio = total_motion_magnitude1 / total_motion_magnitude2
else:
    total_motion_ratio = float('inf')

print(f"上の動きと下の動きの比: {total_motion_ratio:.2f}")

cap.release()
cv2.destroyAllWindows()
