import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

from utils import *
from models import *
import tensorflow as tf


def experiment(save_folder_path, experiment_config, train_model=False):
    print('######################## Experiment details ########################')
    # dataset parameters
    dataset_params = experiment_config['dataset_parameters']
    dataset_filepath = dataset_params['dataset_filepath']
    oscillation_start_idx = dataset_params['oscillation_start_idx']
    unseen_dataset_filepaths = dataset_params['unseen_dataset_filepaths']
    unseen_oscillation_start_idx = dataset_params['unseen_oscillation_start_idx']

    # dataset cropping parameters
    cropping_params = dataset_params['cropping_params']
    crop_x_start = cropping_params['crop_x_start']
    crop_x_end = cropping_params['crop_x_end']
    crop_y_start = cropping_params['crop_y_start']
    crop_y_end = cropping_params['crop_y_end']

    # extracing scaling parameters
    scaling_params = experiment_config['data_scaling_parameters']
    scaling_method = scaling_params['scaling_method']

    # extracting sensor parameters
    sensor_params = experiment_config['sensor_parameters']

    sensor_type = sensor_params['sensor_type']
    down_res = sensor_params['down_res']

    # extracting training parameters
    training_params = experiment_config['training_parameters']

    num_epochs = training_params['num_epochs']
    split_ratio = training_params['split_ratio']
    batch_size = training_params['batch_size']

    # displaying paramters
    print(f'''
Experiment details:
    Experiment folder name: {save_folder_path}

Dataset configuration:
    Dataset filepath: {dataset_filepath}
    Dataset oscillation start index: {oscillation_start_idx}
    Unseen dataset filepaths: {unseen_dataset_filepaths}
    Unseen dataset oscillation start index: {unseen_oscillation_start_idx}
    Unseen dataset cropping configuration:
        crop x start: {crop_x_start}
        crop x end: {crop_x_end}
        crop y start: {crop_y_start}
        crop y end: {crop_y_end}

Preprocessing configuration:
    Data scaling: {scaling_method}

Sensor configuration:
    Sensor type: {sensor_type}
    Down resolution: {down_res}

Training configuration:
    Number of epochs: {num_epochs}
    Train/Test split: {split_ratio}
    Batch size{batch_size}\n''')

    output = {}

    print('######################## Loading flow data ########################')
    y = load_data(dataset_filepath)
    y = np.transpose(y, (0, 2, 1))
    y = y[oscillation_start_idx:,:,:]
    np.random.shuffle(y)
    visualise_loaded_data(y, save_folder_path)
    
    # reshaping (t, m, n, 1)
    y = reshape_for_cnn(y)

    # cropping
    y = y[:, crop_x_start:crop_x_end, crop_y_start:crop_y_end, :]
    y_orig_sample = y[0,:,:,:]

    # resizing
    y = resize_data(y)
    y_resize_sample = y[0,:,:,:]
    visualise_resizing_operation(y_orig_sample, y_resize_sample, save_folder_path)

    # splitting y
    y_train, y_test = split_data(y, split_ratio)

    print('######################## Extracting sensor inputs ########################')
    # creating sensors
    x, _ = get_sensor_data(y, sensor_params)

    # splitting sensors
    x_train, x_test = split_data(x, split_ratio)

    # visualising sensors
    visualise_resolution_reduction(x, y, down_res, save_folder_path)

    print(f'''Extracted unseen data details:
    x shape: {x.shape}
    y shape: {y.shape}\n''')

    if train_model:
        print('######################## Initialising model ########################')
        input_layer_shape = x_train.shape[1:]

        model = scnn(input_layer_shape, down_res)
        model.summary()

        print('######################## Training model ########################')
        # Keras tensorboard callback
        logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,verbose=1)
        time_history_callback = TimeHistory()
        callbacks = [tensorboard_callback, early_stopping_callback, time_history_callback]

        training_start = time.time()
        history = model.fit(
            x = x_train,
            y = y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=(x_test, y_test),
            shuffle=True,
            # verbose=2,
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

        # plotting time history
        times = time_history_callback.times
        plot_time_per_epoch(iterations, times, save_folder_path)
        save_time_per_epoch(times, save_folder_path)
        
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

        # plotting time history
        times, _ = load_time_per_epoch(save_folder_path)
        plot_time_per_epoch(iterations, times, save_folder_path)
        print('######################## Trained model loaded ########################')

    print('######################## Calculating error ########################')
    # predicting, scaling prediction and reshaping prediction
    prediction_start = time.time()
    prediction = model.predict(x_test)
    prediction_end = time.time()

    output['total_prediction_time'] = prediction_end-prediction_start
    output['total_prediction_samples'] = prediction.shape[0]
    print(f'Prediction took {prediction_end-prediction_start} on {prediction.shape[0]} samples')

    # prediction_inv_scaled = unscale_data(prediction_inv_reshaped, scaling_params)
    prediction_inv_scaled = prediction

    # # getting and visualising mask
    # upstream_mask = sensor_params.get('upstream_mask')
    # visualise_upstream_mask(upstream_mask, y_train, sensor_params, save_folder_path)

    # calculating per pixel error and l2 error per image
    # img_per_pixel_abs_error = np.abs(prediction_inv_scaled - y_test) * upstream_mask
    # img_overall_l2_error = calculate_l2_error_norm(prediction_inv_scaled*upstream_mask, y_test*upstream_mask)
    img_per_pixel_abs_error = np.abs(prediction_inv_scaled - y_test)
    img_overall_l2_error = calculate_l2_error_norm(prediction_inv_scaled, y_test)

    output['max_l2_error'] = img_overall_l2_error.max().astype(float)
    output['min_l2_error'] = img_overall_l2_error.min().astype(float)
    output['mean_l2_error'] = img_overall_l2_error.mean().astype(float)

    # visualising error
    visualise_error_across_samples(img_overall_l2_error, save_folder_path)
    visualise_max_error_comparison(x_test, y_test, prediction_inv_scaled, img_per_pixel_abs_error, img_overall_l2_error, save_folder_path)

    # cleaning up
    del y, y_train, y_test
    del x, x_train, x_test
    del prediction, prediction_inv_scaled

    print('######################## Testing against unseen ########################')
    for unseen_obstacle, unseen_dataset_filepath in unseen_dataset_filepaths.items():
        if not os.path.isdir(os.path.join(save_folder_path, unseen_obstacle)):
            os.mkdir(os.path.join(save_folder_path, unseen_obstacle))
        unseen_y = load_data(unseen_dataset_filepath)
        unseen_y = np.transpose(unseen_y, (0, 2, 1))
        unseen_y = unseen_y[unseen_oscillation_start_idx:,:,:]

        visualise_loaded_data(unseen_y, save_folder_path, obstacle=unseen_obstacle)
        
        # reshaping (t, m, n, 1)
        unseen_y = reshape_for_cnn(unseen_y)

        # cropping
        unseen_y = unseen_y[:, crop_x_start:crop_x_end, crop_y_start:crop_y_end, :]

        # resizing
        unseen_y = resize_data(unseen_y)

        print('######################## Extracting unseen sensor inputs ########################')
        # creating sensors
        unseen_x, _ = get_sensor_data(unseen_y, sensor_params)
        # unseen_x = unseen_y[:, ::down_res, ::down_res, :].repeat(down_res, axis=1).repeat(down_res, axis=2)

        # visualising sensors
        visualise_resolution_reduction(unseen_x, unseen_y, down_res, save_folder_path, obstacle=unseen_obstacle)
        
        print(f'''Extracted unseen data details:
        unseen_x shape: {unseen_x.shape}
        unseen_y shape: {unseen_y.shape}\n''')

        print('######################## Calculating unseen error ########################')
        # predicting, scaling prediction and reshaping prediction
        unseen_prediction_start = time.time()
        unseen_prediction = model.predict(unseen_x)
        unseen_prediction_end = time.time()

        output[f'{unseen_obstacle}_total_prediction_time'] = unseen_prediction_end-unseen_prediction_start
        output[f'{unseen_obstacle}_total_prediction_samples'] = unseen_prediction.shape[0]
        print(f'Prediction took {unseen_prediction_end-unseen_prediction_start} on {unseen_prediction.shape[0]} samples')

        # unseen_prediction_inv_scaled = unscale_data(unseen_prediction_inv_reshaped, scaling_params)
        unseen_prediction_inv_scaled = unseen_prediction

        # calculating per pixel error and l2 error per image
        # unseen_img_per_pixel_abs_error = np.abs(unseen_prediction_inv_scaled - unseen_y) * upstream_mask
        # unseen_img_overall_l2_error = calculate_l2_error_norm(unseen_prediction_inv_scaled*upstream_mask, unseen_y*upstream_mask)
        unseen_img_per_pixel_abs_error = np.abs(unseen_prediction_inv_scaled - unseen_y)
        unseen_img_overall_l2_error = calculate_l2_error_norm(unseen_prediction_inv_scaled, unseen_y)

        output[f'{unseen_obstacle}_max_l2_error'] = unseen_img_overall_l2_error.max().astype(float)
        output[f'{unseen_obstacle}_min_l2_error'] = unseen_img_overall_l2_error.min().astype(float)
        output[f'{unseen_obstacle}_mean_l2_error'] = unseen_img_overall_l2_error.mean().astype(float)

        # visualising error
        visualise_error_across_samples(unseen_img_overall_l2_error, save_folder_path, obstacle=unseen_obstacle)
        visualise_max_error_comparison(unseen_x, unseen_y, unseen_prediction_inv_scaled, unseen_img_per_pixel_abs_error, unseen_img_overall_l2_error, save_folder_path, obstacle=unseen_obstacle)

        # cleaning up
        del unseen_y
        del unseen_x
        del unseen_prediction, unseen_prediction_inv_scaled

    return output


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
