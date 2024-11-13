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

# CLAHEと平均値フィルタを組み合わせた輝度調整
def enhance_brightness_and_contrast_with_filter(image, clip_limit=2.0, tile_grid_size=(8, 8), kernel_size=5):
    # 平均値フィルタ（ノイズ除去）
    filtered_image = cv2.blur(image, (kernel_size, kernel_size))

    # CLAHEでコントラスト制限付き適応ヒストグラム平坦化
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(filtered_image)
    
    return enhanced_image

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
    """
    光学フローの動きを矢印で描画する関数
    """
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

def calculate_top_95_percent_movement(mag, min_movement_threshold=10):
    """
    フレーム内で上位95%の動きの大きさを計算し、
    もし上位95%に該当する動きが小さすぎてピクセル数が閾値未満の場合はエラーを出力します。
    """
    valid_magnitudes = mag[mag > 1]  # 動きがあるピクセルのみ

    # 上位95％の動きの大きさを計算
    if valid_magnitudes.size > 0:
        top_95_percent = np.percentile(valid_magnitudes, 95)  # 上位95%の値を計算

        # 上位95%に該当するピクセル数が少なすぎる場合、警告またはエラーを発生
        if valid_magnitudes.size < min_movement_threshold:
            raise ValueError(f"警告: 上位95%の動きに該当するピクセル数が少なすぎます ({valid_magnitudes.size}ピクセル)")

        return top_95_percent
    else:
        raise ValueError("警告: 動きが全く検出されませんでした。")

# 動画を初期化
cap, prvs = initialize_video_capture("assets/echo_data/3_3cm.mp4")

# 動画の最初のフレームで手動でROIを選択
ret, frame = cap.read()
if not ret:
    print("動画の読み込みに失敗しました。")
    exit()

# cv2.selectROIで手動でROIを選択
roi = cv2.selectROI("ROI選択", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("ROI選択")

frame_count = 0

# 合計の上位95%の移動距離を保存する変数
total_top_95_percent_movement = 0.0

# 1ピクセルあたりの距離をコード内で設定 (例えば0.0698 cm)
pixel_to_distance = 0.0698  # 1ピクセルあたりの距離 (cm)

# フラグ：エラーが発生したかどうか
error_occurred = False

# 動画の各フレームに対して処理を行う
while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    # フレームをグレースケールに変換
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # CLAHE + 平均値フィルタを適用
    next_gray = enhance_brightness_and_contrast_with_filter(next_gray)

    # 光学フローを計算
    flow, mag_full = calculate_optical_flow(prvs, next_gray, roi)

    try:
        # フレームごとに上位95%の移動量を計算（ピクセル単位）
        top_95_percent_movement_px = calculate_top_95_percent_movement(mag_full)

        # ピクセル単位から実際の距離（cm）に変換
        top_95_percent_movement_distance = top_95_percent_movement_px * pixel_to_distance

        total_top_95_percent_movement += top_95_percent_movement_distance  # 上位95%の移動距離の合計を更新

        frame_count += 1

        # フレームごとの上位95%の移動量を出力（距離単位で表示）
        print(f"フレーム{frame_count}: 上位95%の移動量: {top_95_percent_movement_distance:.2f} cm")

    except ValueError as e:
        # エラーが発生した場合（動きが小さい、または検出されない場合）
        print(f"フレーム{frame_count}: エラー発生 - {e}")
        error_occurred = True
        break  # エラーが発生したら処理を終了する

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

    prvs = next_gray

# 最終的な上位95%の移動距離の合計を出力
if not error_occurred:
    print(f"ROIの上位95%の移動距離の合計: {total_top_95_percent_movement:.2f} mm")
else:
    print("処理が途中でエラーになりました。")

cap.release()
cv2.destroyAllWindows()

