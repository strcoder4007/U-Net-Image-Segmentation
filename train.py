import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

from data import load_train_data, load_test_data
K.set_image_data_format('channels_last')

def unet_model():
    inputs = Input((512, 512, 1))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    


def train():
    return 

if __name__ == '__main__':
    train()