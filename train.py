import os
import cv2
import numpy as np

from model import Unet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 禁用GPU

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def DataGenerator(file_path, batch_size):
    """
    generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen
    to ensure the transformation for image and mask is the same
    """
    aug_dict = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    aug_dict = dict(horizontal_flip=True,
                    fill_mode='nearest')

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        file_path,
        classes=["images"],
        color_mode="grayscale",
        target_size=(512, 512),
        class_mode=None,
        batch_size=batch_size, seed=1)

    mask_generator = mask_datagen.flow_from_directory(
        file_path,
        classes=["labels"],
        color_mode="grayscale",
        target_size=(512, 512),
        class_mode=None,
        batch_size=batch_size, seed=1)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img = img / 255.
        mask = mask / 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        yield (img, mask)


model = Unet(1, image_size=512)
trainset = DataGenerator("dataset/train", batch_size=2)
model.fit_generator(trainset, steps_per_epoch=100, epochs=1)
model.save_weights("model.h5")

testSet = DataGenerator("dataset/test", batch_size=1)
alpha = 0.3
model.load_weights("model.h5")
if not os.path.exists("./results"): os.mkdir("./results")

for idx, (img, mask) in enumerate(testSet):
    oring_img = img[0]
    # 开始用模型进行预测
    pred_mask = model.predict(img)[0]

    pred_mask[pred_mask > 0.5] = 1
    pred_mask[pred_mask <= 0.5] = 0
    # 如果这里展示的预测结果一片黑，请调整lr，同时注意图片的深度是否为8
    # cv2.imshow('pred_mask', pred_mask)

    # cv2.waitKey()
    img = cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB)
    H, W, C = img.shape
    for i in range(H):
        for j in range(W):
            if pred_mask[i][j][0] <= 0.5:
                img[i][j] = (1 - alpha) * img[i][j] * 255 + alpha * np.array([0, 0, 255])
            else:
                img[i][j] = img[i][j] * 255
    image_accuracy = np.mean(mask == pred_mask)
    image_path = "./results/pred_" + str(idx) + ".png"
    print("=> accuracy: %.4f, saving %s" % (image_accuracy, image_path))
    ret, binary = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
    cv2.imwrite(image_path, binary)

    img1 = cv2.imread(image_path, 0)

    _, binary1 = cv2.threshold(img1, 0.1, 1, cv2.THRESH_BINARY)

    contours, hierarch = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    areas = []  # 记录每一个轮廓所占面积
    result = []  # 记录用于正态分布处理的信息
    for i in range(len(contours)):
        areas.append(cv2.contourArea(contours[i]))  # 计算轮廓所占面积

    result.remove(max(result))  # 移除极端值

    result_shape = np.array(result)

    threshold_min = result_shape.mean() / 4

    threshold_max = result_shape.mean() * 4
    print(areas)
    for x in range(0, len(areas)):
        if areas[x] < threshold_min:
            cv2.drawContours(img1, [contours[x]], 0, 0, -1)
        if areas[x] > threshold_max:
            cv2.drawContours(img1, [contours[x]], 0, 0, -1)

    image_path = "./results/result_" + str(idx) + ".png"
    cv2.imwrite(image_path, img1)  # 保存图片

    cv2.imwrite("./results/origin_%d.png" % idx, oring_img * 255)
    if idx == 18: break
