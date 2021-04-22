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


def srcnn(input_layer_shape, down_res):
    #Model
    input_img = tf.keras.layers.Input(shape=input_layer_shape)

    resize = tf.keras.layers.experimental.preprocessing.Resizing(input_layer_shape[0]*down_res, input_layer_shape[1]*down_res, interpolation='bicubic')(input_img)

    pad = tf.keras.layers.ZeroPadding2D(padding=(6, 6))(resize)

    x1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(9,9), kernel_initializer='glorot_uniform', activation='relu', padding='valid', use_bias=True)(pad)
    x2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform', activation='relu', padding='same', use_bias=True)(x1)
    x_final = tf.keras.layers.Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform', activation='linear', padding='valid', use_bias=True)(x2)

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


def autoencoder(input_layer_shape):
    #Model
    input_img = tf.keras.layers.Input(shape=input_layer_shape)

    x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform', activation='relu', padding='valid')(input_img)
    x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='glorot_uniform', activation='relu', padding='valid')(x1)
    x3 = tf.keras.layers.Conv2D(filters=down_res*down_res, kernel_size=(5,5), kernel_initializer='glorot_uniform', activation='linear',  padding='same')(x2)
    # x4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform', activation='linear', padding='same')(x3)
    x_final = tf.nn.depth_to_space(x3, down_res)

    model = tf.keras.models.Model(input_img, x_final)
    model.compile(optimizer='adam', loss='mse')

    return model


def interpolation(images, down_res, method):
    target_size = [images.shape[1] * down_res, images.shape[2] * down_res]
    print(f'''Bicubic model
    Method: {method}
    Original size of: {images.shape[1:3]}
    Target size of: {target_size}''')
    return tf.image.resize(images, target_size, method=method).numpy()


# class PixelShuffler(Layer):
#     def __init__(self, size=(2, 2), data_format=None, **kwargs):
#         super(PixelShuffler, self).__init__(**kwargs)
#         self.data_format = conv_utils.normalize_data_format(data_format)
#         self.size = conv_utils.normalize_tuple(size, 2, 'size')

#     def call(self, inputs):

#         input_shape = K.int_shape(inputs)
#         if len(input_shape) != 4:
#             raise ValueError('Inputs should have rank ' +
#                              str(4) +
#                              '; Received input shape:', str(input_shape))

#         if self.data_format == 'channels_first':
#             batch_size, c, h, w = input_shape
#             if batch_size is None:
#                 batch_size = -1
#             rh, rw = self.size
#             oh, ow = h * rh, w * rw
#             oc = c // (rh * rw)

#             out = K.reshape(inputs, (batch_size, rh, rw, oc, h, w))
#             out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
#             out = K.reshape(out, (batch_size, oc, oh, ow))
#             return out

#         elif self.data_format == 'channels_last':
#             batch_size, h, w, c = input_shape
#             if batch_size is None:
#                 batch_size = -1
#             rh, rw = self.size
#             oh, ow = h * rh, w * rw
#             oc = c // (rh * rw)

#             out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
#             out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
#             out = K.reshape(out, (batch_size, oh, ow, oc))
#             return out

#     def compute_output_shape(self, input_shape):

#         if len(input_shape) != 4:
#             raise ValueError('Inputs should have rank ' +
#                              str(4) +
#                              '; Received input shape:', str(input_shape))

#         if self.data_format == 'channels_first':
#             height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
#             width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
#             channels = input_shape[1] // self.size[0] // self.size[1]

#             if channels * self.size[0] * self.size[1] != input_shape[1]:
#                 raise ValueError('channels of input and size are incompatible')

#             return (input_shape[0],
#                     channels,
#                     height,
#                     width)

#         elif self.data_format == 'channels_last':
#             height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
#             width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
#             channels = input_shape[3] // self.size[0] // self.size[1]

#             if channels * self.size[0] * self.size[1] != input_shape[3]:
#                 raise ValueError('channels of input and size are incompatible')

#             return (input_shape[0],
#                     height,
#                     width,
#                     channels)

#     def get_config(self):
#         config = {'size': self.size,
#                   'data_format': self.data_format}
#         base_config = super(PixelShuffler, self).get_config()

#         return dict(list(base_config.items()) + list(config.items()))