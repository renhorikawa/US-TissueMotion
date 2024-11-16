import numpy as np
import cv2

def process_video_with_echo_flow_95(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()

    if not ret:
        print("動画の読み込みに失敗しました")
        return

    # 前処理：グレースケール変換とぼかし処理
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)

    # ROIを選択
    x, y, w, h = cv2.selectROI("ROI Selection", old_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("ROI Selection")

    # ROIのマスク作成
    roi_mask = np.zeros(old_gray.shape, dtype=np.uint8)
    roi_mask = cv2.rectangle(roi_mask, (x, y), (x + w, y + h), (255), -1)

    # 特徴点を検出
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=500, qualityLevel=0.1, minDistance=7, blockSize=7, mask=roi_mask)

    if p0 is None:
        print("特徴点の検出に失敗しました")
        cap.release()
        return

    # EchoFlow95用の累積変数
    total_top_95_percent_movement = 0.0

    # アフィンオプティカルフローのパラメータ
    lk_params = dict(winSize=(200, 200), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 変換行列（初期値）
    affine_matrix = np.eye(2, 3, dtype=np.float32)

    # ピクセルからミリメートルへの変換係数
    # ここでは仮に 1ピクセル = 0.1mm としています。実際のケースに合わせて設定してください。
    pixel_to_mm = 0.1  # 1ピクセルあたりの距離（mm）

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        # 光学フロー計算
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # アフィン変換行列の推定（良い特徴点のみで計算）
        if len(good_new) > 0 and len(good_old) > 0:
            affine_matrix, inliers = cv2.estimateAffine2D(good_old, good_new)

            # 変換行列を適用して新しい特徴点を補正
            transformed_points = cv2.transform(good_old.reshape(-1, 1, 2), affine_matrix).reshape(-1, 2)

            # 光学フローのマグニチュードを計算
            dx = transformed_points[:, 0] - good_old[:, 0]
            dy = transformed_points[:, 1] - good_old[:, 1]
            mag = np.sqrt(dx**2 + dy**2)

            # 上位95%の動きの大きさを計算（ピクセル単位）
            top_95_percent_movement_px = np.percentile(mag, 95)

            # ピクセルをミリメートルに変換
            top_95_percent_movement_mm = top_95_percent_movement_px * pixel_to_mm

            # EchoFlow95の累積計算
            total_top_95_percent_movement += top_95_percent_movement_mm

            # デバッグ用：フレームごとの95%の移動量を表示
            print(f"フレームの95%の移動量: {top_95_percent_movement_mm:.2f} mm")

            # トラッキングラインを描画
            for new, old in zip(transformed_points, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
                cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # トラッキング結果の表示
        cv2.imshow('Frame with EchoFlow95', frame)

        # ESCで終了
        key = cv2.waitKey(10)
        if key == 27:
            break

        # 次のフレーム用にデータを更新
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()

    # 最終的な上位95%の移動距離の合計を表示（ミリメートル単位）
    print(f"全フレームにおける上位95%の移動距離の合計: {total_top_95_percent_movement:.2f} mm")

# 動画のパスを指定
video_path = 'assets/echo_data/1_3cm.mp4'
process_video_with_echo_flow_95(video_path)

