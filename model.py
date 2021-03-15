from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D


def Unet(num_class, image_size):
    inputs = Input(shape=[image_size, image_size, 1])  # [256,256,1]
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # [128,128,64]
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # [64,64,128]
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    drop3 = Dropout(0.5)(conv3)  # [64,64,256]
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)  # [32,32,256]

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)  # [32,32,512]
    drop4 = Dropout(0.5)(conv4)  # [32,32,512]
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) #[16,16,512]

    # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    # drop5 = Dropout(0.5)(conv5)#[16,16,1024]
    #
    # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))#[32,32,512]
    # merge6 = concatenate([drop4,up6], axis = 3)#[32,32,1024]
    # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)#[32,32,512]

    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))  # [64,64,256]
    merge7 = concatenate([conv3, up7], axis=3)  # [64,64,512]
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))  # [128,128,128]
    merge8 = concatenate([conv2, up8], axis=3)  # [128,128,256]
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))  # [256,256,64]
    merge9 = concatenate([conv1, up9], axis=3)  # [256,256,128]
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)  # [256,256,2]
    conv10 = Conv2D(num_class, 1, activation='sigmoid')(conv9)  # [256,256,1]
    model = Model(inputs=inputs, outputs=conv10)
    # 这里的lr根据自己的数据集调整，选取不当容易导致预测结果全黑的情况
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model
