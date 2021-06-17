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
    unseen_cropping_params = dataset_params['unseen_cropping_params']
    unseen_crop_x_start = unseen_cropping_params['unseen_crop_x_start']
    unseen_crop_x_end = unseen_cropping_params['unseen_crop_x_end']
    unseen_crop_y_start = unseen_cropping_params['unseen_crop_y_start']
    unseen_crop_y_end = unseen_cropping_params['unseen_crop_y_end']

    # extracing scaling parameters
    scaling_params = experiment_config['data_scaling_parameters']
    scaling_method = scaling_params['scaling_method']

    # extracting sensor parameters
    sensor_params = experiment_config['sensor_parameters']

    sensor_type = sensor_params['sensor_type']
    sensor_config_str = '\n'.join([f'    {k}: {v}' for k,v in sensor_params.items()])

    # extracting training parameters
    training_params = experiment_config['training_parameters']

    num_epochs = training_params['num_epochs']
    split_ratio = training_params['split_ratio']
    learning_rate = training_params['learning_rate']
    weight_decay = training_params['weight_decay']
    learning_rate_change = training_params['learning_rate_change']
    weight_decay_change  = training_params['weight_decay_change']
    epoch_update = training_params['epoch_update']
    early_stopping = training_params['early_stopping']

    # displaying paramters
    print(f'''
Experiment details:
    Experiment folder name: {save_folder_path}

Dataset configuration:
    Dataset filepath: {dataset_filepath}
    Dataset oscillation start index: {oscillation_start_idx}
    Unseen dataset filepath: {unseen_dataset_filepaths}
    Unseen dataset oscillation start index: {unseen_oscillation_start_idx}
    Unseen dataset cropping configuration:
        crop x start: {unseen_crop_x_start}
        crop x end: {unseen_crop_x_end}
        crop y start: {unseen_crop_y_start}
        crop y end: {unseen_crop_y_end}

Preprocessing configuration:
    Data scaling: {scaling_method}

Sensor configuration:
    Sensor type: {sensor_type}
    {sensor_config_str}

Training configuration:
    Number of epochs: {num_epochs}
    Train/Test split: {split_ratio}
    AdamW optimiser config:
        Learning rate: {learning_rate}
        Weight decay: {weight_decay}
        Learnign rate change: {learning_rate_change}
        Weight decay change: {weight_decay_change}
        Epoch for update: {epoch_update}\n''')

    output = {}

    print('######################## Loading flow data ########################')
    y = load_data(dataset_filepath)
    y = y[oscillation_start_idx:,:,:]
    y = np.transpose(y, (0, 2, 1))
    np.random.seed(42)
    np.random.shuffle(y)
    visualise_loaded_data(y, save_folder_path)
    
    print('######################## Reshaping and scaling data ########################')
    # splitting
    y_train, y_test, m, n = split_data(y, split_ratio)

    # Scaling
    y_train_scaled, y_test_scaled, scaling_params = rescale_data(y_train, y_test, scaling_params=scaling_params)

    # reshaping
    y_train_reshaped = reshape_for_ann(y_train_scaled)
    y_test_reshaped = reshape_for_ann(y_test_scaled)
    print(f'''Reshaped data details:
    train_data_reshaped has the shape: {y_train_reshaped.shape}
    test_data_reshaped has the shape: {y_test_reshaped.shape}\n''')

    print('######################## Extracting sensor inputs ########################')
    # Updating sensor params
    sensor_params['m'] = m
    sensor_params['n'] = n

    # Getting sensor locations
    sensors, sensor_params = get_sensor_data(y_train_reshaped, sensor_params)
    sensors_test, _ = get_sensor_data(y_test_reshaped, sensor_params)
    output['sensor_num'] = sensor_params['sensor_num']

    # reshaping sensors
    sensors = reshape_for_ann(sensors)
    sensors_test = reshape_for_ann(sensors_test)

    print(f'''Extracted sensor data details:
    sensors has the shape: {sensors.shape}
    sensors_test has the shape: {sensors_test.shape}\n''')

    # visualising sensors
    visualise_extracted_sensor_locations(y_train, sensor_params, save_folder_path)
    visualise_extracted_sensor_outputs(sensors, save_folder_path)

    if train_model:
        print('######################## Initialising model ########################')
        input_layer_shape = sensors.shape[1:]
        output_layer_size = y_train_reshaped.shape[-1]

        model = original_shallow_decoder(input_layer_shape, output_layer_size, learning_rate, weight_decay)
        model.summary()

        print('######################## Training model ########################')
        # Learning rate scheduler callback
        def scheduler(epoch, lr):
            if epoch % epoch_update:
                return lr
            else:
                return lr * learning_rate_change
        learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # Keras tensorboard callback
        logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        # Early stopping callback
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        
        if early_stopping:
            callbacks = [tensorboard_callback, learning_rate_callback, early_stopping_callback]
        else:
            callbacks = [tensorboard_callback, learning_rate_callback]

        training_start = time.time()
        history = model.fit(
            x = sensors,
            y = y_train_reshaped,
            epochs=num_epochs,
            callbacks=callbacks,
            validation_data=(sensors_test, y_test_reshaped),
            shuffle=True,
            verbose=2,
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
    prediction = model.predict(sensors_test)
    prediction_end = time.time()

    output['total_prediction_time'] = prediction_end-prediction_start
    output['total_prediction_samples'] = prediction.shape[0]
    print(f'Prediction took {prediction_end-prediction_start} on {prediction.shape[0]} samples')

    prediction_inv_reshaped = reshape_for_img(prediction, sensor_params)
    prediction_inv_scaled = unscale_data(prediction_inv_reshaped, scaling_params)

    # getting and visualising mask
    upstream_mask = sensor_params.get('upstream_mask')
    visualise_upstream_mask(upstream_mask, y_train, sensor_params, save_folder_path)

    # calculating per pixel error and l2 error per image
    img_per_pixel_abs_error = np.abs(prediction_inv_scaled - y_test) * upstream_mask
    img_overall_l2_error = calculate_l2_error_norm(prediction_inv_scaled*upstream_mask, y_test*upstream_mask)

    output['max_l2_error'] = img_overall_l2_error.max()
    output['min_l2_error'] = img_overall_l2_error.min()
    output['mean_l2_error'] = img_overall_l2_error.mean()

    # visualising error
    visualise_error_across_samples(img_overall_l2_error, save_folder_path)
    visualise_max_error_comparison(y_test, prediction_inv_scaled, img_per_pixel_abs_error, img_overall_l2_error, save_folder_path)

    # predicting with POD plus
    print('######################## Predicting with pod plus ########################')
    pod_model = POD(plus=True)
    pod_model.summary()

    sensor_locations = sensor_params['pivots']

    pod_training_start = time.time()
    pod_model.fit(
        x = sensors.reshape(sensors.shape[0], -1),
        y = y_train.reshape(y_train.shape[0], -1),
        sensor_locations = sensor_locations,
        alpha=1e-8,
    )
    pod_training_end = time.time()

    output['total_training_time_pod'] = pod_training_end-pod_training_start
    print(f'POD training took {pod_training_end-pod_training_start}s')

    pod_prediction_start = time.time()
    pod_prediction = pod_model.predict(sensors_test.reshape(sensors_test.shape[0],-1))
    pod_prediction_end = time.time()

    output['total_prediction_time_pod'] = pod_prediction_end-pod_prediction_start
    output['total_prediction_samples_pod'] = pod_prediction.shape[0]
    print(f'Prediction took {pod_prediction_end-pod_prediction_start} on {pod_prediction.shape[0]} samples')


    pod_prediction = pod_prediction.reshape(pod_prediction.shape[0],m,n)
    pod_prediction_inv_scaled = unscale_data(pod_prediction, scaling_params)
    
    # calculating per pixel error and l2 error per image
    pod_img_per_pixel_abs_error = np.abs(pod_prediction_inv_scaled - y_test) * upstream_mask
    pod_img_overall_l2_error = calculate_l2_error_norm(pod_prediction_inv_scaled*upstream_mask, y_test*upstream_mask)

    output['max_l2_error_pod'] = pod_img_overall_l2_error.max()
    output['min_l2_error_pod'] = pod_img_overall_l2_error.min()
    output['mean_l2_error_pod'] = pod_img_overall_l2_error.mean()

    # visualising error
    if not os.path.isdir(os.path.join(save_folder_path, 'POD')):
            os.mkdir(os.path.join(save_folder_path, 'POD'))
    visualise_error_across_samples(pod_img_overall_l2_error, save_folder_path, obstacle='POD')
    visualise_max_error_comparison(y_test, pod_prediction_inv_scaled, pod_img_per_pixel_abs_error, pod_img_overall_l2_error, save_folder_path, obstacle='POD')

    # cleaning up
    del y, y_train, y_test, y_train_scaled, y_test_scaled, y_train_reshaped, y_test_reshaped
    del sensors, sensors_test
    del prediction, prediction_inv_reshaped, prediction_inv_scaled
    del pod_prediction, pod_prediction_inv_scaled

    print('######################## Testing against unseen ########################')
    for unseen_obstacle, unseen_dataset_filepath in unseen_dataset_filepaths.items():
        if not os.path.isdir(os.path.join(save_folder_path, unseen_obstacle)):
            os.mkdir(os.path.join(save_folder_path, unseen_obstacle))
        unseen_y = load_data(unseen_dataset_filepath)
        unseen_y = unseen_y[unseen_oscillation_start_idx:,:,:]
        unseen_y = np.transpose(unseen_y, (0, 2, 1))
        unseen_y = unseen_y[:, unseen_crop_x_start:unseen_crop_x_end, unseen_crop_y_start:unseen_crop_y_end]

        visualise_loaded_data(unseen_y, save_folder_path, obstacle=unseen_obstacle)

        # Scaling
        unseen_y_scaled, _, _ = rescale_data(unseen_y, None, scaling_params=scaling_params)

        # reshaping
        unseen_y_reshaped = reshape_for_ann(unseen_y_scaled)
        print(f'''Reshaped unseen data details:
        unseen_data_reshaped has the shape: {unseen_y_scaled.shape}\n''')

        # getting sensor outputs
        unseen_sensors, _ = get_sensor_data(unseen_y_reshaped, sensor_params)

        # reshaping unseen sensors
        unseen_sensors = reshape_for_ann(unseen_sensors)

        print(f'''Extracted unseen sensor data details:
        unseen_sensors has the shape: {unseen_sensors.shape}\n''')

        # visualising unseen sensors
        visualise_extracted_sensor_locations(unseen_y, sensor_params, save_folder_path, obstacle=unseen_obstacle)
        visualise_extracted_sensor_outputs(unseen_sensors, save_folder_path, obstacle=unseen_obstacle)

        print('######################## Calculating unseen error ########################')
        # predicting, scaling prediction and reshaping prediction
        unseen_prediction_start = time.time()
        unseen_prediction = model.predict(unseen_sensors)
        unseen_prediction_end = time.time()

        output[f'{unseen_obstacle}_total_prediction_time'] = unseen_prediction_end-unseen_prediction_start
        output[f'{unseen_obstacle}_total_prediction_samples'] = unseen_prediction.shape[0]
        print(f'Prediction took {unseen_prediction_end-unseen_prediction_start} on {unseen_prediction.shape[0]} samples')

        unseen_prediction_inv_reshaped = reshape_for_img(unseen_prediction, sensor_params)
        unseen_prediction_inv_scaled = unscale_data(unseen_prediction_inv_reshaped, scaling_params)

        # calculating per pixel error and l2 error per image
        unseen_img_per_pixel_abs_error = np.abs(unseen_prediction_inv_scaled - unseen_y) * upstream_mask
        unseen_img_overall_l2_error = calculate_l2_error_norm(unseen_prediction_inv_scaled*upstream_mask, unseen_y*upstream_mask)

        output[f'{unseen_obstacle}_max_l2_error'] = unseen_img_overall_l2_error.max()
        output[f'{unseen_obstacle}_min_l2_error'] = unseen_img_overall_l2_error.min()
        output[f'{unseen_obstacle}_mean_l2_error'] = unseen_img_overall_l2_error.mean()

        # visualising error
        visualise_error_across_samples(unseen_img_overall_l2_error, save_folder_path, obstacle=unseen_obstacle)
        visualise_max_error_comparison(unseen_y, unseen_prediction_inv_scaled, unseen_img_per_pixel_abs_error, unseen_img_overall_l2_error, save_folder_path, obstacle=unseen_obstacle)

        # cleaning up
        del unseen_y, unseen_y_scaled, unseen_y_reshaped
        del unseen_sensors
        del unseen_prediction, unseen_prediction_inv_reshaped, unseen_prediction_inv_scaled

    return output