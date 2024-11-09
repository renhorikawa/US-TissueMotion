import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_contrast_histogram_equalization(image_path):
    # 画像を読み込む
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # グレースケールで読み込み

    # ヒストグラム均等化
    enhanced_image = cv2.equalizeHist(image)

    # ヒストグラムの計算
    hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_enhanced = cv2.calcHist([enhanced_image], [0], None, [256], [0, 256])

    # 画像とヒストグラムを表示
    plt.figure(figsize=(12, 6))

    # 元の画像と均等化後の画像を表示
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    # ヒストグラムを表示
    plt.subplot(2, 1, 2)
    plt.plot(hist_original, color='blue', label='Original Histogram')
    plt.plot(hist_enhanced, color='red', label='Enhanced Histogram')
    plt.title('Histograms')
    plt.xlim([0, 256])
    plt.legend()

    plt.tight_layout()
    plt.show()

# メイン処理
input_image_path = 'output_images/frame_1_1.png'
enhance_contrast_histogram_equalization(input_image_path)

