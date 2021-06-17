import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

import skfmm

from utils import *
from models import *
import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array, load_img


def experiment(save_folder_path, experiment_config, train_model=False):
    print('######################## Experiment details ########################')
    # dataset parameters
    dataset_params = experiment_config['dataset_parameters']
    x_dataset_filepath = dataset_params['x_dataset_filepath']
    y_dataset_filepath = dataset_params['y_dataset_filepath']
    train_ratio = dataset_params['train_ratio']
    val_ratio = dataset_params['val_ratio']
    test_ratio = dataset_params['test_ratio']
    flipped = dataset_params['flipped']
    use_sdf = dataset_params['use_sdf']
    
    # extracting training parameters
    training_params = experiment_config['training_parameters']
    num_epochs = training_params['num_epochs']
    batch_size = training_params['batch_size']

    # displaying paramters
    print(f'''
Experiment details:
    Experiment folder name: {save_folder_path}

Dataset configuration:
    X Dataset filepath: {x_dataset_filepath}
    Y Dataset filepath: {y_dataset_filepath}
    Train ratio: {train_ratio}
    Validation ratio: {val_ratio}
    Test ratio: {test_ratio}
    Flipped: {flipped}
    Use SDF: {use_sdf}

Training configuration:
    Number of epochs: {num_epochs}
    Batch size: {batch_size}\n''')

    output = {}

    print('######################## Loading flow data ########################')
    x = load_data(x_dataset_filepath)
    y = load_data(y_dataset_filepath)
    visualise_loaded_data(x, y, save_folder_path)

    print(f'x data: min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
    print(f'y data: min: {y.min()}, max: {y.max()}, mean: {y.mean()}')

    if flipped:
        print('######################## Flipping data ########################')
        x_flipped = np.flip(x, 1)
        y_flipped = y
        x = np.concatenate([x_flipped, x])
        y = np.concatenate([y_flipped, y])
        print(f'The shape of x flipped data (t, m, n) is {x.shape}')
        print(f'The shape of y flipped data (t, m, n) is {y.shape}')

    if use_sdf:
        print('######################## Generating SDF for data ########################')
        x[x==0] = -1
        sdf = np.zeros(x.shape)
        for i in range(x.shape[0]):
            sdf[i,:,:,0] = skfmm.distance(x[i,:,:,0], dx=1/50)
        x = sdf
        print(f'x data: min: {x.min()}, max: {x.max()}, mean: {x.mean()}')

    print('######################## Splitting and augmenting data ########################')
    # splitting
    x_train, x_val, x_test = split_data(x, train_ratio, val_ratio, test_ratio)
    y_train, y_val, y_test = split_data(y, train_ratio, val_ratio, test_ratio)

    if train_model:
        print('######################## Initialising model ########################')
        input_layer_shape = x_train.shape[1:]

        model = cnn_viquerat(input_layer_shape)
        model.summary()

        print('######################## Training model ########################')
        # Keras tensorboard callback
        logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',verbose=0,
            patience=10
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            f'{save_folder_path}/best.h5',
            monitor='val_loss',
            mode='min',
            verbose=0,
            save_best_only=True,
            save_weights_only=False
        )
        time_history_callback = TimeHistory()
        callbacks = [tensorboard_callback, early_stopping_callback, checkpoint_callback, time_history_callback]

        training_start = time.time()
        history = model.fit(
            x = x_train,
            y = y_train,
            epochs=num_epochs,
            callbacks=callbacks,
            validation_data=(x_val, y_val),
        )
        training_end = time.time()
        output['total_training_time'] = training_end-training_start
        output['total_epochs'] = len(history.epoch)
        print(f'Training on {len(history.epoch)} took {training_end-training_start}s')

        print('######################## Saving model ########################')
        # plotting error history
        iterations, train_loss, val_loss = extract_train_val_error(history)
        plot_train_val_error(iterations, train_loss, val_loss, save_folder_path)
        save_train_val_error(iterations, train_loss, val_loss, save_folder_path)
        
        model.save(f'{save_folder_path}/model.h5')
        print('######################## Trained model saved ########################')
    else:
        print('######################## Initialising model ########################')
        model_path = f'{save_folder_path}/model.h5'
        model = tf.keras.models.load_model(model_path)
        model.summary()

        # plotting error history
        iterations, train_loss, val_loss, _ = load_train_val_error(save_folder_path)
        plot_train_val_error(iterations, train_loss, val_loss, save_folder_path)
        print('######################## Trained model loaded ########################')

    print('######################## Calculating error ########################')
    # predicting, scaling prediction and reshaping prediction
    prediction_start = time.time()
    prediction = model.predict(x_test)
    prediction_end = time.time()

    output['total_prediction_time'] = prediction_end-prediction_start
    output['total_prediction_samples'] = prediction.shape[0]
    print(f'Prediction took {prediction_end-prediction_start} on {prediction.shape[0]} samples')

    abs_error = calculate_absolute_error(y_test, prediction)
    rel_error = calculate_relative_error(y_test, prediction)

    output['max_abs_error'] = abs_error.max().astype(float)
    output['min_abs_error'] = abs_error.min().astype(float)
    output['mean_abs_error'] = abs_error.mean().astype(float)

    output['max_rel_error'] = rel_error.max().astype(float)
    output['min_rel_error'] = rel_error.min().astype(float)
    output['mean_rel_error'] = rel_error.mean().astype(float)

    # visualising error
    visualise_error_across_samples(rel_error, save_folder_path)
    visualise_max_error_comparison(y_test, prediction, rel_error, x_test, save_folder_path)

    # cleaning up
    del x, x_train, x_val, x_test
    del y, y_train, y_val, y_test
    del prediction, abs_error, rel_error

    return output


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)