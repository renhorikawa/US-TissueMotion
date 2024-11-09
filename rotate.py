import cv2
import numpy as np

input_video_path = 'assets/a.mp4'
output_video_path = 'assets/rotated_a.mp4'

angle = 270

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: 動画ファイルを開けませんでした。")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

center = (frame_width // 2, frame_height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_width, frame_height))
    out.write(rotated_frame)

cap.release()
out.release()

print(f"動画の回転が完了しました。出力ファイル: {output_video_path}")

