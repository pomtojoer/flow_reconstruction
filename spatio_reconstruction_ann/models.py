import tensorflow as tf
import tensorflow_addons as tfa


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