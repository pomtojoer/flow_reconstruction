import tensorflow as tf
import tensorflow_addons as tfa


def cnn_viquerat(input_layer_shape):
    #Model
    input_img = tf.keras.layers.Input(shape=input_layer_shape)

    conv1a = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(input_img)
    conv1b = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(conv1a)
    pool1 = tf.keras.layers.MaxPooling2D((2,2))(conv1b)

    conv2a = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(pool1)
    conv2b = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(conv2a)
    pool2 = tf.keras.layers.MaxPooling2D((2,2))(conv2b)

    conv3a = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(pool2)
    conv3b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(conv3a)
    pool3 = tf.keras.layers.MaxPooling2D((2,2))(conv3b)

    conv4a = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(pool3)
    conv4b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(conv4a)
    pool4 = tf.keras.layers.MaxPooling2D((2,2))(conv4b)

    conv5a = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(pool4)
    conv5b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(conv5a)
    pool5 = tf.keras.layers.MaxPooling2D((2,2))(conv5b)

    flat = tf.keras.layers.Flatten()(pool5)
    fc_layer = tf.keras.layers.Dense(64, activation='relu')(flat)
    x_final = tf.keras.layers.Dense(1, activation='linear')(fc_layer)

    model = tf.keras.models.Model(input_img, x_final)

    optimiser = tf.keras.optimizers.Adam(lr=1e-3, decay=5e-3)

    model.compile(optimizer=optimiser, loss='mse')

    return model