import cv2
import numpy as np

# 读图
img = cv2.imread('results/result_18.png', 0)
# 设置核
kernel = np.ones((3, 3), np.uint8)
# 开运算(先腐蚀后膨胀)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imwrite('opening/opening_18.png', opening)
