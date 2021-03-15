import cv2
import numpy as np

img = cv2.imread(r'juchi/0.png', 0)

_, binary = cv2.threshold(img, 0.1, 1, cv2.THRESH_BINARY)

contours, hierarch = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

areas = []  # 记录每一个轮廓所占面积
result = []  # 记录用于正态分布处理的信息

for i in range(len(contours)):
    areas.append(cv2.contourArea(contours[i]))  # 计算轮廓所占面积
    if cv2.contourArea(contours[i]) != 0:
        result.append(cv2.contourArea(contours[i]))

result_shape = np.array(result)

threshold_min = result_shape.mean() / 4

for x in range(0, len(areas)):
    if areas[x] < threshold_min:
        cv2.drawContours(img, [contours[x]], 0, 0, -1)

cv2.imwrite(r'opening/0.png', img)  # 保存图片
