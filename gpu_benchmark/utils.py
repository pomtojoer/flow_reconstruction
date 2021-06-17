import numpy as np
import pandas as pd

import os
import math

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean

import tensorflow as tf

# -----------------------------------------------------------
# Data loading functions
# -----------------------------------------------------------

def load_data(filepath):
    full_filepath = os.path.join('..', 'data', filepath)
    data = np.load(full_filepath)

    print(f'The shape of the data (t, m, n) is {data.shape}')

    return data


def visualise_loaded_data(data, save_folder, sample_times=[1, 10, 50, 100, 150], obstacle=None):
    print('Visualising loaded data')
    
    assert len(sample_times) > 1, 'sample_times length needs to be at least 2'

    number_of_samples = len(sample_times)
    fig, axs = plt.subplots(1, number_of_samples, figsize=(14,6), facecolor='white')
    for idx, sample in enumerate(sample_times):
        minmax = np.nanmax(np.abs(data[sample,:,:])) * 0.65
        axs[idx].imshow(data[sample,:,:], cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
        axs[idx].set_title(f't={sample}')
    plt.tight_layout()
    if obstacle is not None:
        plt.savefig(f'{save_folder}/{obstacle}/{obstacle}_loaded_data_visualisation.png', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(f'{save_folder}/loaded_data_visualisation.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
    

# -----------------------------------------------------------
# Data augmentation functions
# -----------------------------------------------------------

def split_data(data, ratio):
    t = data.shape[0]
    cutoff = int(t*ratio)
    
    train_data = data[:cutoff,:,:]
    test_data = data[cutoff:,:,:]
    
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    assert train_data.shape[0] == cutoff
    assert test_data.shape[0] == t-cutoff

    print(f'''Split data details:
    train_data has {train_data.shape[0]} examples
    test_data has {test_data.shape[0]} examples\n''')
    
    return train_data, test_data


def resize_data(data, output_size=(256, 128)):
    resize_model = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(output_size[0], output_size[1])])
    resized_data = resize_model(data).numpy()

    print(f'''Resized data details:
    original data has shape {data.shape}
    resized data has shape {resized_data.shape}\n''')
    
    return resize_model(data).numpy()


def visualise_resizing_operation(y_orig_sample, y_resize_sample, save_folder_path, obstacle=None):
    fig, axs = plt.subplots(1, 2, facecolor='white', edgecolor='k', figsize=(8, 4))
    minmax = np.nanmax(np.abs(y_orig_sample)) * 0.65
    axs[0].imshow(y_orig_sample, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    axs[0].set_title(f'Original image of shape {y_orig_sample.shape}')
    axs[1].imshow(y_resize_sample, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    axs[1].set_title(f'Resized image of shape {y_resize_sample.shape}')
    plt.tight_layout()
    if obstacle is not None:
        plt.savefig(f'{save_folder_path}/{obstacle}/{obstacle}_resizing_operation_comparison.png', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(f'{save_folder_path}/resizing_operation_comparison.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()



# def rescale_data(x_train, x_test=None, scaling_params=None):
#     method = scaling_params.get('scaling_method', None)

#     if method=='normalise':
#         x_min = scaling_params.get('x_min')
#         x_max = scaling_params.get('x_max')
#         if x_min is None:
#             x_min = x_train.min(axis=0)
#         if x_max is None:
#             x_max = x_train.max(axis=0)

#         x_train = (x_train - x_min) / (x_max - x_min)
#         if x_test is not None:
#             x_test = (x_test - x_min) / (x_max - x_min)

#         scaling_params = {
#             'scaling_method': method,
#             'x_min': x_min,
#             'x_max': x_max    
#         }
    
#     elif method=='normalise_balanced':
#         x_mean = scaling_params.get('x_mean')
#         x_min = scaling_params.get('x_min')
#         x_max = scaling_params.get('x_max')

#         if x_mean is None:
#             x_mean = x_train.mean(axis=0)
#         if x_min is None:
#             x_min = x_train.min(axis=0)
#         if x_max is None:
#             x_max = x_train.max(axis=0)

#         x_train = (x_train - x_mean) / (x_max - x_min)
#         if x_test is not None:
#             x_test = (x_test - x_mean) / (x_max - x_min)

#         scaling_params = {
#             'scaling_method': method,
#             'x_mean': x_mean,
#             'x_min': x_min,
#             'x_max': x_max    
#         }
    
#     elif method=='standardise':
#         x_mean = scaling_params.get('x_mean')
#         x_std = scaling_params.get('x_std')

#         if x_mean is None:
#             x_mean = x_train.mean(axis=0)
#         if x_std is None:
#             x_std = x_train.std(axis=0)

#         out_train = np.zeros((x_train.shape))
#         x_train = np.divide((x_train - x_mean), x_std, out=out_train, where=x_std!=0)
#         if x_test is not None:
#             out_test = np.zeros((x_test.shape))
#             x_test = np.divide((x_test - x_mean), x_std, out=out_test, where=x_std!=0)

#         scaling_params = {
#             'scaling_method': method,
#             'x_mean': x_mean,
#             'x_std': x_std    
#         }

#     elif method=='center':
#         x_mean = scaling_params.get('x_mean')

#         if x_mean is None:
#             x_mean = x_train.mean(axis=0)

#         x_train = x_train - x_mean
#         if x_test is not None:
#             x_test = x_test - x_mean

#         scaling_params = {
#             'scaling_method': method,
#             'x_mean': x_mean,
#         }

#     elif method=='center_all':
#         x_mean = scaling_params.get('x_mean')

#         if x_mean is None:
#             x_mean = x_train.mean()

#         x_train = x_train - x_mean
#         if x_test is not None:
#             x_test = x_test - x_mean

#         scaling_params = {
#             'scaling_method': method,
#             'x_mean': x_mean,
#         }

#     else:
#         print('Unknown scaling type. Returning data unscaled')

#         scaling_params = {
#             'scaling_method': 'unscaled',
#         }

#     print(f'''Scaled data details:
#     train_data_rescaled has min: {x_train.min():.4f}, max: {x_train.max():.4f}, mean: {x_train.mean():.4f}
#     test_data_rescaled has min: {x_test.min() if x_test is not None else 0:.4f}, max: {x_test.max() if x_test is not None else 0:.4f}, mean: {x_test.mean() if x_test is not None else 0:.4f}\n''')

#     return x_train, x_test, scaling_params

# def unscale_data(x, scaling_params):
#     method = scaling_params['scaling_method']

#     if method=='normalise':
#         x_min = scaling_params['x_min']
#         x_max = scaling_params['x_max']

#         x = x * (x_max - x_min) + x_min
    
#     elif method=='normalise_balanced':
#         x_mean = scaling_params['x_mean']
#         x_min = scaling_params['x_min']
#         x_max = scaling_params['x_max']

#         x = x * (x_max - x_min) + x_mean
    
#     elif method=='standardise':
#         x_mean = scaling_params['x_mean']
#         x_std = scaling_params['x_std']
        
#         x = x * x_std + x_mean

#     elif method=='center' or method=='center_all':
#         x_mean = scaling_params['x_mean']

#         x = x + x_mean

#     else:
#         pass

#     return x

def reshape_for_cnn(x):
    return x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    

# def reshape_for_img(x, sensor_params):
#     m = sensor_params.get('m')
#     n = sensor_params.get('n')
    
#     return x.reshape(x.shape[0],m,n)

# -----------------------------------------------------------
# Sensor extraction functions
# -----------------------------------------------------------

def get_sensor_data(y, sensor_params):
    sensor_type = sensor_params['sensor_type']
    if sensor_type == 'equal':
        return _equal_size_input(y, sensor_params)

    if sensor_type == 'reduce':
        return _reduced_size_input(y, sensor_params)

    else:
        return _equal_size_input(y, sensor_params)

def _equal_size_input(y, sensor_params):
    down_res = sensor_params.get('down_res', 1)
    output = y[:, ::down_res, ::down_res, :].repeat(down_res, axis=1).repeat(down_res, axis=2)
    return output, sensor_params

def _reduced_size_input(y, sensor_params):
    down_res = sensor_params.get('down_res', 1)
    output = y[:, ::down_res, ::down_res, :]
    return output, sensor_params

def visualise_resolution_reduction(x, y, down_res, save_folder_path, samples=[0], obstacle=None):
    fig, axs = plt.subplots(len(samples), 2, facecolor='white', edgecolor='k', figsize=(8, 4*len(samples)))
    for idx, sample in enumerate(samples):
        # setting background
        x_sample = x[sample, :, :]
        y_sample = y[sample, :, :]

        minmax = np.nanmax(np.abs(y_sample)) * 0.65

        if len(samples) > 1:
            ax = axs[idx]
        else:
            ax = axs
        
        ax[0].imshow(x_sample, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
        ax[0].set_title(f'1/{down_res} resolution')
        ax[1].imshow(y_sample, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
        ax[1].set_title('original resolution')

        # ax.grid(False)
        # ax.axis('off')

    plt.tight_layout()
    if obstacle is not None:
        plt.savefig(f'{save_folder_path}/{obstacle}/{obstacle}_resolution_reduction_visualisation.png', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(f'{save_folder_path}/resolution_reduction_visualisation.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()

# -----------------------------------------------------------
# train error plots functions
# -----------------------------------------------------------

def extract_train_val_error(history):
    iterations = history.epoch
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    return iterations, train_loss, val_loss


def plot_train_val_error(iterations, train_loss, val_loss, save_folder_path):
    save_file = f'{save_folder_path}/error_per_iteration.png'

    fig = plt.figure(figsize=(7.9,4.7), facecolor='white', edgecolor='k')
    plt.plot(iterations, train_loss, label='train error')
    plt.plot(iterations, val_loss, label='validation error')
    plt.title('Plot of error against iteration')
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(save_file, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()


    save_file_log = f'{save_folder_path}/log_error_per_iteration.png'

    fig = plt.figure(figsize=(7.9,4.7), facecolor='white', edgecolor='k')
    plt.plot(iterations, train_loss, label='train error')
    plt.plot(iterations, val_loss, label='validation error')
    plt.yscale("log")
    plt.title('Plot of log error against iteration')
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(save_file_log, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()

    return save_file, save_file_log


def save_train_val_error(iterations, train_loss, val_loss, save_folder_path):
    save_file = f'{save_folder_path}/training_epoch_data.csv'

    data = {
        'epoch':iterations,
        'train_loss':train_loss,
        'val_loss':val_loss,
    }

    df = pd.DataFrame(data)
    df.to_csv(save_file, index=False)

    return save_file


def load_train_val_error(save_folder_path):
    save_file = f'{save_folder_path}/training_epoch_data.csv'

    df =  pd.read_csv(save_file)

    iterations = df['epoch'].to_numpy()
    train_loss = df['train_loss'].to_numpy()
    val_loss = df['val_loss'].to_numpy()

    return iterations, train_loss, val_loss, save_file


def plot_time_per_epoch(iterations, times, save_folder_path):
    fig = plt.figure(figsize=(7.9,4.7), facecolor='white', edgecolor='k')
    plt.plot(iterations, times)
    plt.title('Plot of time taken [s] per epoch')
    plt.ylabel('Time taken [s]')
    plt.xlabel('Epochs')
    plt.savefig(f'{save_folder_path}/time_per_epoch.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()


def save_time_per_epoch(times, save_folder_path):
    save_file = f'{save_folder_path}/training_epoch_data.csv'
    df = pd.read_csv(save_file)
    df['times'] = times
    df.to_csv(save_file, index=False)

    return save_file


def load_time_per_epoch(save_folder_path):
    save_file = f'{save_folder_path}/training_epoch_data.csv'

    df =  pd.read_csv(save_file)

    times = df['times'].to_numpy()

    return times, save_file

# -----------------------------------------------------------
# prediction error functions
# -----------------------------------------------------------

def calculate_l2_error_norm(actual, prediction, upstream_mask=None):
    actual = actual.reshape(actual.shape[0],1,-1)
    prediction = prediction.reshape(prediction.shape[0],1,-1)
    if upstream_mask is not None:
        actual = actual * upstream_mask
        prediction = prediction * upstream_mask
    return np.linalg.norm(actual-prediction, axis=-1,) / np.linalg.norm(actual, axis=-1,)


# def visualise_upstream_mask(upstream_mask, y, sensor_params, save_folder_path):
#     y_sample = y[0,:,:]
#     minmax = np.nanmax(np.abs(y_sample)) * 0.65
#     fig, axs = plt.subplots(3, 1, figsize=(8,14), facecolor='white', edgecolor='k')
#     axs[0].imshow(upstream_mask.T, cmap='gray')
#     axs[0].set_title('upstream binary mask')
#     axs[1].imshow(y_sample.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
#     axs[1].set_title('actual sample')
#     axs[2].imshow((y_sample*upstream_mask).T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
#     axs[2].set_title('masked sample')
#     plt.tight_layout()
#     plt.savefig(f'{save_folder_path}/upstream_mask.png', facecolor=fig.get_facecolor(), edgecolor='none')
#     plt.show()


def visualise_error_across_samples(error, save_folder_path, samples=[0], obstacle=None):
    print(f'Max error in all samples is {error.max()*100:.4f}%')
    print(f'Min error in all samples is {error.min()*100:.4f}%')
    print(f'Mean error in all samples is {error.mean()*100:.4f}%')

    fig, axs = plt.subplots(len(samples), 1, facecolor='white', edgecolor='k', figsize=(7.9, 4.7*len(samples)))
    for idx, sample in enumerate(samples):
        if len(samples) > 1:
            ax = axs[idx]
        else:
            ax = axs

        if idx < len(samples)-1:
            error_sample = error[sample:samples[idx+1]]
        else:
            error_sample = error[sample:]
        
        ax.plot(range(error_sample.shape[0]), error_sample[:]*100)
        ax.set_title(f'L2 error per sample. Max:{error_sample.max()*100:.4f}%, Min:{error_sample.max()*100:.4f}%, Mean:{error_sample.mean()*100:.4f}%')
        ax.set_ylabel('L2 error percentage, %')
        ax.set_xlabel('Sample number')

    if obstacle is not None:
        plt.savefig(f'{save_folder_path}/{obstacle}/{obstacle}_error_per_sample.png', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(f'{save_folder_path}/error_per_sample.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()


def visualise_max_error_comparison(x, y_actual, y_prediction, error, l2_error, save_folder_path, samples=[0], obstacle=None):
    max_error_sample = np.argmax(l2_error)

    x_sample = x[max_error_sample,:,:]
    y_actual_sample = y_actual[max_error_sample,:,:]
    y_prediction_sample = y_prediction[max_error_sample,:,:]
    error_sample = error[max_error_sample, :, :]
    l2_error_sample = l2_error[max_error_sample,0]

    print(f'Plotting comparison plot for max L2 error sample, sample {max_error_sample}, L2 error {l2_error_sample*100:.4f}%')
    # # initialising plots
    # _, m, n, _ = y_actual.shape

    # x = np.arange(0, m, 1)
    # y = np.arange(0, n, 1)
    # mX, mY = np.meshgrid(x, y)
    minmax = np.nanmax(np.abs(y_actual_sample)) * 0.65

    # Plotting
    fig, axs = plt.subplots(1, 4, facecolor="white",  edgecolor='k', figsize=(16,4))
    fig.suptitle('Output comparison')

    axs[0].imshow(x_sample, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    # axs[0].contourf(mX, mY, x_sample, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    axs[0].set_title(f'input. Sample num ={max_error_sample}')
    axs[0].axis('off')

    axs[1].imshow(y_actual_sample, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    # axs[1].contourf(mX, mY, y_actual_sample, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    axs[1].set_title('actual')        
    axs[1].axis('off')

    axs[2].imshow(y_prediction_sample, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    # axs[2].contourf(mX, mY, y_prediction_sample, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    axs[2].set_title('prediction')        
    axs[2].axis('off')

    im2 = axs[3].imshow(error_sample, cmap='gray_r', interpolation='none', vmin=0, vmax=error_sample.max())
    axs[3].set_title(f'Absolute difference. Overall L2 error = {l2_error_sample*100:.4f}%')
    axs[3].axis('off')
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax)
    # fig.colorbar(im2, ax=axs[3], orientation='horizontal')

    fig.tight_layout()
    if obstacle is not None:
        plt.savefig(f'{save_folder_path}/{obstacle}/{obstacle}_max_error_comparison.png')
    else:
        plt.savefig(f'{save_folder_path}/max_error_comparison.png')
    plt.show()


# -----------------------------------------------------------
# prediction error functions
# -----------------------------------------------------------

