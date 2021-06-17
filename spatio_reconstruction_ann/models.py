import tensorflow as tf
import tensorflow_addons as tfa

from sklearn import linear_model
import numpy as np

def original_shallow_decoder(input_layer_shape, output_layer_size, learning_rate, weight_decay):
    # Model
    model = tf.keras.models.Sequential([
        # Layer 1
        tf.keras.layers.Input(input_layer_shape),
        tf.keras.layers.Dense(40, activation='relu', kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        
        # Layer 2
        tf.keras.layers.Dense(45, activation='relu', kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        
        # Output layer
        tf.keras.layers.Dense(output_layer_size),
    ])

    # Optimiser
    optimiser = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    # Defining losses
    losses = tf.keras.losses.MeanSquaredError()

    # Defining metrics
    metrics = tf.keras.metrics.MeanAbsoluteError()

    model.compile(optimizer=optimiser, loss=losses, metrics=metrics)

    return model


class POD:
    def __init__(self, plus=False):
        self.u = None
        self.s = None
        self.v = None
        self.plus = plus
        
    def summary(self):
        text = (
            'Model: POD\n'
            '_________________________________________________________________\n'
            'Steps            Explanation\n'
            '=================================================================\n'
            'SVD              POD decomposition is carried out via singular\n'
            '                 value decomposition (SVD) - np.linalg.svd(y,0)\n'
            '_________________________________________________________________\n'
            'linear_coeff     Solution to the least square problem between\n'
            '                 actual sensors and decomposed sensors with the\n'
            '                 minimum L2 norm is solved via the Moore-Penrose\n'
            '                 pseudo-inverse\n'
            '_________________________________________________________________\n'
            'Prediction       Linear combination of reduced POD modes with\n'
            '                 previously calculated linear coefficients\n'
            '=================================================================\n'
            'Access modes: POD.v\n'
            'Access singular value: POD.s\n'
            f'POD plus - {self.plus}\n'
        )
        print(text)
        
    def fit(self, x, y, sensor_locations, alpha=0):
        assert y.ndim == 2
        
        self.sensor_locations = sensor_locations
        self.n_sensors, = self.sensor_locations.shape
        
        self.x_train = x
        self.y_train = y
        self.alpha = alpha
        
        print(f'Input shape {self.x_train.shape}')
        print(f'Output shape {self.y_train.shape}')
        
        u, s, vt = np.linalg.svd(self.y_train.T, 0)
        self.u = u
        self.s = s
        self.v = vt.T
        
        if self.plus:
            self.p = self.x_train.T.dot(np.linalg.pinv(self.y_train.T).dot(self.u[:,0:self.n_sensors]))
        else:
            self.p = self.u[self.sensor_locations, 0:self.n_sensors]
        
        
    def predict(self, x):
        assert x.ndim == 2
        
        if self.alpha == 0:
            self.reg = linear_model.LinearRegression(fit_intercept=False, normalize=False)
        else:
            self.reg = linear_model.Ridge(alpha=self.alpha, fit_intercept=False, normalize=False)
        
        self.reg.fit(self.p, x.T)
        
        self.linear_coef = self.reg.coef_.T
        return (self.u[:,0:self.n_sensors].dot(self.linear_coef)).T