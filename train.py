import os
import numpy as np
import tensorflow as tf

from skimage.transform import resize
from skimage.io import imsave

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

from data import load_train_data, load_test_data
K.set_image_data_format('channels_last')

def unet_model():
    inputs = Input((512, 512, 1))

    # Contracting layers
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(maxpool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(maxpool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(maxpool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    maxpool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(maxpool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    # Expansive Layers
    conv_transpose1 = Conv2DTranspose(512, (2, 2), padding='same')(conv5)
    # dont quite understand this, concatenate should make the channel dimension 1024 making it impossible to convolve with 512 channel dims
    conv_cat1 = Concatenate(axis=3)([conv_transpose1, conv4])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_cat1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    conv_transpose2 = Conv2DTranspose(256, (2, 2), padding='same')(conv6)
    conv_cat2 = Concatenate(axis=3)([conv_transpose2, conv3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_cat2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    conv_transpose3 = Conv2DTranspose(128, (2, 2), padding='same')(conv7)
    conv_cat3 = Concatenate(axis=3)([conv_transpose3, conv2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_cat3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    conv_transpose4 = Conv2DTranspose(64, (2, 2), padding='same')(conv8)
    conv_cat4 = Concatenate(axis=3)([conv_transpose4, conv1])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_cat4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    # last layer
    conv10 = Conv2D(2, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def dice_coef(y_true, y_pred):
    y_true_f = Flatten(y_true)
    y_pred_f = Flatten(y_pred)
    intersection = tf.keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.sum(y_true_f) + tf.keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def preprocess_images(imgs):
    processed_imgs = np.ndarray((imgs.shape[0], 512, 512), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        processed_imgs[i] = resize(imgs[i], (512, 512), preserve_range=True)

    processed_imgs = processed_imgs[..., np.newaxis]
    return processed_imgs


def normalize_images(imgs, mask):
    imgs = imgs.astype('float32')
    mean = np.mean(imgs)
    std = np.std(imgs)

    imgs = imgs-mean
    imgs = imgs/std

    mask = mask.astype('float32')
    mask /= 255.

    return imgs, mask



def train():
    train_imgs, train_imgs_mask = load_train_data()

    train_imgs = preprocess_images(train_imgs)
    train_imgs_mask = preprocess_images(train_imgs_mask)

    train_imgs, train_imgs_mask = normalize_images(train_imgs, train_imgs_mask)

    model = unet_model()

    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    model.fit(train_imgs, 
              train_imgs_mask, 
              batch_size=32, 
              nb_epoch=20, 
              verbose=1, 
              shuffle=True, 
              validation_split=0.2, 
              callbacks=[model_checkpoint])
    
    



    return 

if __name__ == '__main__':
    train()