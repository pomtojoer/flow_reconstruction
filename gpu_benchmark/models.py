import tensorflow as tf
import tensorflow_addons as tfa

import tensorflow.keras.utils as conv_utils
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

def dsc_ms(input_layer_shape):
    #Model
    input_img = tf.keras.layers.Input(shape=input_layer_shape)

    #Down sampled skip-connection model
    down_1 = tf.keras.layers.MaxPooling2D((8,8),padding='same')(input_img)
    x1 = tf.keras.layers.Conv2D(32, (3,3),activation='relu', padding='same')(down_1)
    x1 = tf.keras.layers.Conv2D(32, (3,3),activation='relu', padding='same')(x1)
    x1 = tf.keras.layers.UpSampling2D((2,2))(x1)

    down_2 = tf.keras.layers.MaxPooling2D((4,4),padding='same')(input_img)
    x2 = tf.keras.layers.Concatenate()([x1,down_2])
    x2 = tf.keras.layers.Conv2D(32, (3,3),activation='relu', padding='same')(x2)
    x2 = tf.keras.layers.Conv2D(32, (3,3),activation='relu', padding='same')(x2)
    x2 = tf.keras.layers.UpSampling2D((2,2))(x2)

    down_3 = tf.keras.layers.MaxPooling2D((2,2),padding='same')(input_img)
    x3 = tf.keras.layers.Concatenate()([x2,down_3])
    x3 = tf.keras.layers.Conv2D(32, (3,3),activation='relu', padding='same')(x3)
    x3 = tf.keras.layers.Conv2D(32, (3,3),activation='relu', padding='same')(x3)
    x3 = tf.keras.layers.UpSampling2D((2,2))(x3)

    x4 = tf.keras.layers.Concatenate()([x3,input_img])
    x4 = tf.keras.layers.Conv2D(32, (3,3),activation='relu', padding='same')(x4)
    x4 = tf.keras.layers.Conv2D(32, (3,3),activation='relu', padding='same')(x4)

    #Multi-scale model (Du et al., 2018)
    layer_1 = tf.keras.layers.Conv2D(16, (5,5),activation='relu', padding='same')(input_img)
    x1m = tf.keras.layers.Conv2D(8, (5,5),activation='relu', padding='same')(layer_1)
    x1m = tf.keras.layers.Conv2D(8, (5,5),activation='relu', padding='same')(x1m)

    layer_2 = tf.keras.layers.Conv2D(16, (9,9),activation='relu', padding='same')(input_img)
    x2m = tf.keras.layers.Conv2D(8, (9,9),activation='relu', padding='same')(layer_2)
    x2m = tf.keras.layers.Conv2D(8, (9,9),activation='relu', padding='same')(x2m)

    layer_3 = tf.keras.layers.Conv2D(16, (13,13),activation='relu', padding='same')(input_img)
    x3m = tf.keras.layers.Conv2D(8, (13,13),activation='relu', padding='same')(layer_3)
    x3m = tf.keras.layers.Conv2D(8, (13,13),activation='relu', padding='same')(x3m)

    x_add = tf.keras.layers.Concatenate()([x1m,x2m,x3m,input_img])
    x4m = tf.keras.layers.Conv2D(8, (7,7),activation='relu',padding='same')(x_add)
    x4m = tf.keras.layers.Conv2D(3, (5,5),activation='relu',padding='same')(x4m)

    x_final = tf.keras.layers.Concatenate()([x4,x4m])
    x_final = tf.keras.layers.Conv2D(input_layer_shape[-1], (3,3),padding='same')(x_final)

    model = tf.keras.models.Model(input_img, x_final)
    model.compile(optimizer='adam', loss='mse')

    return model


def scnn(input_layer_shape, down_res):
    #Model
    input_img = tf.keras.layers.Input(shape=input_layer_shape)

    x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), kernel_initializer='glorot_uniform', activation='tanh', padding='same')(input_img)
    x2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), kernel_initializer='glorot_uniform', activation='tanh', padding='same')(x1)
    x3 = tf.keras.layers.Conv2D(filters=down_res*down_res, kernel_size=(5,5), kernel_initializer='glorot_uniform', activation='linear',  padding='same')(x2)
    # x4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform', activation='linear', padding='same')(x3)
    x_final = tf.nn.depth_to_space(x3, down_res)

    model = tf.keras.models.Model(input_img, x_final)
    model.compile(optimizer='adam', loss='mse')

    return model