import numpy as np
import pandas as pd

import os
import math

import matplotlib.pyplot as plt
import cmocean

# -----------------------------------------------------------
# Data loading functions
# -----------------------------------------------------------

def load_data(filepath):
    full_filepath = os.path.join('..', 'data', filepath)
    data = np.load(full_filepath)

    print(f'The shape of the data (t, m, n) is {data.shape}')

    return data


def visualise_loaded_data(x_data, y_data, save_folder, sample_times=[1, 10, 50, 100, 150], obstacle=None):
    print('Visualising loaded data')
    
    assert len(sample_times) > 1, 'sample_times length needs to be at least 2'

    number_of_samples = len(sample_times)
    fig, axs = plt.subplots(1, number_of_samples, figsize=(14,6), facecolor='white')
    for idx, sample in enumerate(sample_times):
        axs[idx].imshow(x_data[sample,:,:])
        axs[idx].set_title(f'Idx: {sample}, Cd={y_data[sample,]}')
    plt.tight_layout()
    if obstacle is not None:
        plt.savefig(f'{save_folder}/{obstacle}/{obstacle}_loaded_data_visualisation.png', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(f'{save_folder}/loaded_data_visualisation.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
    

# -----------------------------------------------------------
# Data augmentation functions
# -----------------------------------------------------------

def split_data(dataset, train_size, valid_size, tests_size):
    # Check sizes
    if ((train_size + valid_size + tests_size) != 1.0):
        print('Error in split_dataset')
        print('The sum of the three provided sizes must be 1.0')
        exit()

    # Compute sizes
    n_data     = dataset.shape[0]
    train_size = math.floor(n_data*train_size)
    valid_size = math.floor(n_data*valid_size) + train_size
    tests_size = math.floor(n_data*tests_size) + valid_size

    # Split
    if (dataset.ndim == 1):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size],
                           dataset[train_size:valid_size],
                           dataset[valid_size:])

    if (dataset.ndim == 2):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size,         :],
                           dataset[train_size:valid_size,:],
                           dataset[valid_size:,          :])

    if (dataset.ndim == 3):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size,         :,:],
                           dataset[train_size:valid_size,:,:],
                           dataset[valid_size:,          :,:])

    if (dataset.ndim == 4):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size,         :,:,:],
                           dataset[train_size:valid_size,:,:,:],
                           dataset[valid_size:,          :,:,:])

    print(f'''Split data details:
    train has {dataset_train.shape[0]} examples
    validation has {dataset_valid.shape[0]} examples
    test has {dataset_tests.shape[0]} examples\n''')

    return dataset_train, dataset_valid, dataset_tests


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


# -----------------------------------------------------------
# prediction error functions
# -----------------------------------------------------------

def calculate_absolute_error(actual, prediction):
    return np.abs(prediction - actual)

def calculate_relative_error(actual, prediction, eps=1e-6):
    return np.abs(prediction - actual) / np.abs(actual + eps)


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
        ax.set_title(f'Relative error per sample. Max:{error_sample.max()*100:.4f}%, Min:{error_sample.max()*100:.4f}%, Mean:{error_sample.mean()*100:.4f}%')
        ax.set_ylabel('Relative error percentage, %')
        ax.set_xlabel('Sample number')

    if obstacle is not None:
        plt.savefig(f'{save_folder_path}/{obstacle}/{obstacle}_error_per_sample.png', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(f'{save_folder_path}/error_per_sample.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()


def visualise_max_error_comparison(actual, prediction, error, shape, save_folder_path, samples=5, obstacle=None):
    error = error.reshape(error.shape[0])
    parts = np.argpartition(error, -samples)[-samples:]
    max_err_idxs = parts[np.argsort((-error)[parts])]

    fig, axs = plt.subplots(samples, facecolor="white",  edgecolor='k', figsize=(4, 4*samples))
    fig.suptitle('Max prediction error\n')

    for i, ax in enumerate(axs):
        max_err_idx = max_err_idxs[i]

        actual_sample = actual[max_err_idx,0]
        prediction_sample = prediction[max_err_idx,0]
        error_sample = error[max_err_idx]
        shape_sample = shape[max_err_idx,:,:,:]

        ax.imshow(shape_sample)
        ax.set_title(f'Actual: {actual_sample:.4f}\nPrediction: {prediction_sample:.4f}\nError: {error_sample*100:.4f}%')

    fig.tight_layout()
    if obstacle is not None:
        plt.savefig(f'{save_folder_path}/{obstacle}/{obstacle}_max_error_comparison.png')
    else:
        plt.savefig(f'{save_folder_path}/max_error_comparison.png')
    plt.show()


# -----------------------------------------------------------
# prediction error functions
# -----------------------------------------------------------

# def save_outputs()