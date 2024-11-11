import numpy as np
import cv2

# 実装の参考　http://www.beechtreetech.com/opencv-exercises-in-python
# cv2を使うように変更

updatelock = False # トラックバー処理中のロックフラグ
windowname = 'movie' # Windowの名前
trackbarname = 'current' # トラックバーの名前

ESC_KEY = 0x1b

cap = cv2.VideoCapture('assets/sample3.mp4')

# トラックバーを動かしたときに呼び出されるコールバック関数の定義
def onTrackbarSlide(pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    updatelock = True

# 名前付きWindowを定義する
cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)

# AVIファイルのフレーム数を取得する
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

class Sample:
    def __init__(self):
        self.interval = 10

    def run(self):
        # フレーム数が1以上ならトラックバーにセットする
        if (frames > 0):
            cv2.createTrackbar(trackbarname, windowname, 0, frames, onTrackbarSlide)

        # AVIファイルを開いている間は繰り返し（最後のフレームまで読んだら終わる）
        while(cap.isOpened()):
            # トラックバー更新中は描画しない
            if (updatelock):
                continue
            # １フレーム読む
            ret, frame = cap.read()
            # 読めなかったら抜ける
            if ret == False:
                break
            # 画面に表示
            cv2.imshow(windowname,frame)
            # 現在のフレーム番号を取得
            curpos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # トラックバーにセットする（コールバック関数が呼ばれる）
            cv2.setTrackbarPos(trackbarname, windowname, curpos)

            key = cv2.waitKey(self.interval)
            # "Esc"キー押下で終了
            if key == ESC_KEY:
                break
            # "s"キー押下で一時停止
            elif key == ord("q"):
                self.interval = 0
            elif key == ord("r"):
                self.interval = 10

Sample().run()

# AVIファイルを解放
cap.release()
# Windowを閉じる
cv2.destroyAllWindows()
