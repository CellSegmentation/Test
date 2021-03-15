from PIL import Image
import cv2
import numpy as np

img_path = "opening_1_5.5.png"  # 获取图片路径
img = Image.open(img_path)
img_sp = cv2.imread(img_path)
sp = img_sp.shape  # 读取图片长宽

# 第一次插值、降噪
img = img.resize((sp[1] * 2, sp[0] * 2), Image.BILINEAR)  # 三次样条插值，图像长宽放大为2倍
img = np.array(img)
img_gaosi = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯降噪，设置高斯核
img = Image.fromarray(img_gaosi)  # 转换回数组，以便numpy可读取
img = img.resize((sp[1] * 2, sp[0] * 2), Image.ANTIALIAS)  # 保持图像品质缩略

# 第二次插值、降噪
img = img.resize((sp[1] * 4, sp[0] * 4), Image.BILINEAR)  # 三次样条插值，图像长宽放大为4倍
img = np.array(img)
img_gaosi = cv2.GaussianBlur(img, (5, 5), 0)
img = Image.fromarray(img_gaosi)
img = img.resize((sp[1] * 4, sp[0] * 4), Image.ANTIALIAS)

# 第三次插值、降噪
img = img.resize((sp[1] * 16, sp[0] * 16), Image.BILINEAR)  # 三次样条插值，图像长宽放大为16倍
img = np.array(img)
img_gaosi = cv2.GaussianBlur(img, (5, 5), 0)
img = Image.fromarray(img_gaosi)
img = img.resize((sp[1] * 16, sp[0] * 16), Image.ANTIALIAS)

# img = img.convert('L')  # 图像二值化
# pixdata = img.load()
# w, h = img.size
# for y in range(h):
#     for x in range(w):
#         if pixdata[x, y] < 127:
#             pixdata[x, y] = 0
#         else:
#             pixdata[x, y] = 255
img.save('5.5.1.png')
