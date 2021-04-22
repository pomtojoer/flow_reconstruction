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
    t, m, n = data.shape
    cutoff = int(t*ratio)
    
    train_data = data[:cutoff,:,:]
    test_data = data[cutoff:,:,:]
    
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    assert train_data.shape == (cutoff, m, n)
    assert test_data.shape == (t-cutoff, m, n)

    print(f'''Split data details:
    train_data has {train_data.shape[0]} examples
    test_data has {test_data.shape[0]} examples\n''')
    
    return train_data, test_data, m, n


def rescale_data(x_train, x_test=None, scaling_params=None):
    method = scaling_params.get('scaling_method', None)

    if method=='normalise':
        x_min = scaling_params.get('x_min')
        x_max = scaling_params.get('x_max')
        if x_min is None:
            x_min = x_train.min(axis=0)
        if x_max is None:
            x_max = x_train.max(axis=0)

        x_train = (x_train - x_min) / (x_max - x_min)
        if x_test is not None:
            x_test = (x_test - x_min) / (x_max - x_min)

        scaling_params = {
            'scaling_method': method,
            'x_min': x_min,
            'x_max': x_max    
        }
    
    elif method=='normalise_balanced':
        x_mean = scaling_params.get('x_mean')
        x_min = scaling_params.get('x_min')
        x_max = scaling_params.get('x_max')

        if x_mean is None:
            x_mean = x_train.mean(axis=0)
        if x_min is None:
            x_min = x_train.min(axis=0)
        if x_max is None:
            x_max = x_train.max(axis=0)

        x_train = (x_train - x_mean) / (x_max - x_min)
        if x_test is not None:
            x_test = (x_test - x_mean) / (x_max - x_min)

        scaling_params = {
            'scaling_method': method,
            'x_mean': x_mean,
            'x_min': x_min,
            'x_max': x_max    
        }
    
    elif method=='standardise':
        x_mean = scaling_params.get('x_mean')
        x_std = scaling_params.get('x_std')

        if x_mean is None:
            x_mean = x_train.mean(axis=0)
        if x_std is None:
            x_std = x_train.std(axis=0)

        out_train = np.zeros((x_train.shape))
        x_train = np.divide((x_train - x_mean), x_std, out=out_train, where=x_std!=0)
        if x_test is not None:
            out_test = np.zeros((x_test.shape))
            x_test = np.divide((x_test - x_mean), x_std, out=out_test, where=x_std!=0)

        scaling_params = {
            'scaling_method': method,
            'x_mean': x_mean,
            'x_std': x_std    
        }

    elif method=='center':
        x_mean = scaling_params.get('x_mean')

        if x_mean is None:
            x_mean = x_train.mean(axis=0)

        x_train = x_train - x_mean
        if x_test is not None:
            x_test = x_test - x_mean

        scaling_params = {
            'scaling_method': method,
            'x_mean': x_mean,
        }

    elif method=='center_all':
        x_mean = scaling_params.get('x_mean')

        if x_mean is None:
            x_mean = x_train.mean()

        x_train = x_train - x_mean
        if x_test is not None:
            x_test = x_test - x_mean

        scaling_params = {
            'scaling_method': method,
            'x_mean': x_mean,
        }

    else:
        print('Unknown scaling type. Returning data unscaled')

        scaling_params = {
            'scaling_method': 'unscaled',
        }

    print(f'''Scaled data details:
    train_data_rescaled has min: {x_train.min():.4f}, max: {x_train.max():.4f}, mean: {x_train.mean():.4f}
    test_data_rescaled has min: {x_test.min() if x_test is not None else 0:.4f}, max: {x_test.max() if x_test is not None else 0:.4f}, mean: {x_test.mean() if x_test is not None else 0:.4f}\n''')

    return x_train, x_test, scaling_params

def unscale_data(x, scaling_params):
    method = scaling_params['scaling_method']

    if method=='normalise':
        x_min = scaling_params['x_min']
        x_max = scaling_params['x_max']

        x = x * (x_max - x_min) + x_min
    
    elif method=='normalise_balanced':
        x_mean = scaling_params['x_mean']
        x_min = scaling_params['x_min']
        x_max = scaling_params['x_max']

        x = x * (x_max - x_min) + x_mean
    
    elif method=='standardise':
        x_mean = scaling_params['x_mean']
        x_std = scaling_params['x_std']
        
        x = x * x_std + x_mean

    elif method=='center' or method=='center_all':
        x_mean = scaling_params['x_mean']

        x = x + x_mean

    else:
        pass

    return x

def reshape_for_ann(x):
    return x.reshape(x.shape[0], 1, -1)
    

def reshape_for_img(x, sensor_params):
    m = sensor_params.get('m')
    n = sensor_params.get('n')
    
    return x.reshape(x.shape[0],m,n)

# -----------------------------------------------------------
# Sensor extraction functions
# -----------------------------------------------------------

def get_sensor_data(x, sensor_params):
    sensor_type = sensor_params['sensor_type']
    
    if sensor_type == 'wall':
        return _set_sensor_wall(x, sensor_params)

    if sensor_type == 'line':
        return _set_sensor_line(x, sensor_params)

    if sensor_type == 't':
        return _set_sensor_t(x, sensor_params)

    if sensor_type == 'patch':
        return _set_sensor_patch(x, sensor_params)
        
    if sensor_type == 'resolution_reduction':
        return _set_sensor_resolution_reduction(x, sensor_params)

def _set_sensor_wall(x, sensor_params):
    if x.ndim == 2:
        n_snapshots, n_pix = x.shape
    elif x.ndim == 3:
        n_snapshots, _ ,n_pix = x.shape
    
    sensor_num = sensor_params.get('sensor_num', 5)
    pivots = sensor_params.get('pivots')
    sensor_distribution = sensor_params.get('sensor_distribution', 'equispace')

    if pivots is None:
        # extracting input parameters
        random_seed = sensor_params.get('random_seed', 12345)
        m = sensor_params.get('m', 384)
        n = sensor_params.get('n', 199)
        origin_x = sensor_params.get('origin_x', 0)
        origin_y = sensor_params.get('origin_y', 99)
        rad = sensor_params.get('rad', 36)
        
        # initialising mask
        mask = np.zeros((m,n))

        # determining sensor distribution
        if sensor_distribution == 'front_equispace':
            theata = np.linspace(-np.pi/2, np.pi/2, sensor_num)
            x_cord = -np.round(rad * np.cos(theata)) + origin_x
        elif sensor_distribution == 'rear_equispace':
            theata = np.linspace(-np.pi/2, np.pi/2, sensor_num)
            x_cord = np.round(rad * np.cos(theata)) + origin_x
        elif sensor_distribution == 'equispace':
            theata = np.linspace(0, 2*np.pi, sensor_num+1)
            x_cord = np.round(rad * np.cos(theata)) + origin_x
        elif sensor_distribution == 'random':
            theata = np.linspace(0, 2*np.pi, 300)
            x_cord = np.round(rad * np.cos(theata)) + origin_x
        
        # creating sensor coordinates
        y_cord = np.round(rad * np.sin(theata)) + origin_y
        cords = np.vstack((x_cord,y_cord)).T
        cords = np.unique(cords, axis=0)
        
        # ensure no 'wrap around' on image
        idx = cords[:,0] > 0
        cords = cords[idx,:]

        # randomising choice
        if sensor_distribution == 'random':
            np.random.seed(random_seed)
            idx = np.random.choice(range(cords.shape[0]), sensor_num, False)
            cords = cords[idx,:]
        
        # extracting sensor inputs
        cords = np.int64(cords)
        mask[cords[:,0], cords[:,1]] = 1
        pivots = np.where(mask.reshape(-1) == 1)
        pivots = np.asarray(pivots).ravel()

        upstream_mask = np.ones((m,n))
        upstream_mask[:cords[:,0].min(),:] = 0

        sensor_params['pivots'] = pivots
        sensor_params['upstream_mask'] = upstream_mask

    if x.ndim == 2:
        sensors = x[:, pivots].reshape(n_snapshots, sensor_num)
    else:
        sensors = x[:, :, pivots].reshape(n_snapshots, sensor_num)

    return sensors, sensor_params


def _set_sensor_line(x, sensor_params):
    if x.ndim == 2:
        n_snapshots, n_pix = x.shape
    elif x.ndim == 3:
        n_snapshots, _ ,n_pix = x.shape

    sensor_num = sensor_params.get('sensor_num', 5)
    pivots = sensor_params.get('pivots')

    if pivots is None:
        sensor_distribution = sensor_params.get('sensor_distribution', 'equispace')
        m = sensor_params.get('m', 384)
        n = sensor_params.get('n', 199)
        spanwise_orientation = sensor_params.get('spanwise', True)
        line_start = sensor_params.get('line_start', 0)
        line_end = sensor_params.get('line_end', n)
        if spanwise_orientation:
            origin = sensor_params.get('origin_x', 0)
        else:
            origin = sensor_params.get('origin_y', int(n/2))

        mask = np.zeros((m,n))
        if spanwise_orientation:
            y_cord = np.arange(line_start, line_end)
            x_cord = np.full(y_cord.shape, origin)
        else:
            x_cord = np.arange(line_start, line_end)
            y_cord = np.full(x_cord.shape, origin)
        cords = np.vstack((x_cord,y_cord)).T
        cords = np.unique(cords, axis=0)

        if sensor_distribution == 'equispace':
            idx = np.linspace(0, cords.shape[0]-1, num=sensor_num, dtype=int)
        elif sensor_distribution == 'random':
            random_seed = sensor_params.get('random_seed', 12345)
            np.random.seed(random_seed)
            idx = np.random.choice(range(cords.shape[0]), sensor_num, False)
        
        cords = np.int64(cords[idx,:])

        mask[cords[:,0], cords[:,1]] = 1
        pivots = np.where(mask.reshape(-1) == 1)
        pivots = np.asarray(pivots).ravel()

        upstream_mask = np.ones((m,n))
        if spanwise_orientation:
            upstream_mask[:line_start,:] = 0
        else:
            upstream_mask[:origin,:] = 0

        sensor_params['pivots'] = pivots
        sensor_params['upstream_mask'] = upstream_mask

        if spanwise_orientation:
            dist_from_rear = (origin-75)*1/50
            total_length = (cords[-1,1]-cords[0,1])*1/50
            separation_length = (cords[1,1]-cords[0,1])*1/50
        else:
            dist_from_rear = (line_start-75)*1/50
            total_length = (cords[-1,0]-cords[0,0])*1/50
            separation_length = (cords[1,0]-cords[0,0])*1/50

        print(f'''Physically:
        Distance from obstacle rear: {dist_from_rear} unit lengths
        Total line length: {total_length} unit lengths
        Sensor separation: {separation_length} unit lengths
        ''')

    if x.ndim == 2:
        sensors = x[:, pivots].reshape(n_snapshots, sensor_num)
    else:
        sensors = x[:, :, pivots].reshape(n_snapshots, sensor_num)

    return sensors, sensor_params


def _set_sensor_t(x, sensor_params):
    if x.ndim == 2:
        n_snapshots, n_pix = x.shape
    elif x.ndim == 3:
        n_snapshots, _ ,n_pix = x.shape

    sensor_num = sensor_params.get('sensor_num')
    pivots = sensor_params.get('pivots')

    if pivots is None:
        sensor_distribution = sensor_params.get('sensor_distribution', 'equispace')

        sensor_num_spanwise = sensor_params.get('sensor_num_spanwise', 3)
        sensor_num_streamwise = sensor_params.get('sensor_num_streamwise', 3)

        m = sensor_params.get('m', 384)
        n = sensor_params.get('n', 199)

        line_start_spanwise = sensor_params.get('line_start_spanwise', 0)
        line_end_spanwise = sensor_params.get('line_end_spanwise', m)
        line_end_streamwise = sensor_params.get('line_end_streamwise', n)

        origin_x = sensor_params.get('origin_x', 0)
        origin_y = sensor_params.get('origin_y', int(n/2))

        mask = np.zeros((m,n))

        spanwise_y_cord = np.arange(line_start_spanwise, line_end_spanwise)
        spanwise_x_cord = np.full(spanwise_y_cord.shape, origin_x)
        
        streamwise_x_cord = np.arange(origin_x, line_end_streamwise)
        streamwise_y_cord = np.full(streamwise_x_cord.shape, origin_y)

        spanwise_cord = np.vstack((spanwise_x_cord,spanwise_y_cord)).T
        streamwise_cord = np.vstack((streamwise_x_cord,streamwise_y_cord)).T
        
        spanwise_cord = np.unique(spanwise_cord, axis=0)
        streamwise_cord = np.unique(streamwise_cord, axis=0)

        if sensor_distribution == 'equispace':
            spanwise_idx = np.linspace(0, spanwise_cord.shape[0]-1, num=sensor_num_spanwise, dtype=int)
            streamwise_idx = np.linspace(0, streamwise_cord.shape[0]-1, num=sensor_num_streamwise, dtype=int)
        elif sensor_distribution == 'random':
            random_seed = sensor_params.get('random_seed', 12345)
            np.random.seed(random_seed)
            spanwise_idx = np.random.choice(range(spanwise_cord.shape[0]), sensor_num_spanwise, False)
            streamwise_idx = np.random.choice(range(streamwise_cord.shape[0]), sensor_num_streamwise, False)
        
        spanwise_cord = np.int64(spanwise_cord[spanwise_idx,:])
        streamwise_cord = np.int64(streamwise_cord[streamwise_idx,:])

        mask[spanwise_cord[:,0], spanwise_cord[:,1]] = 1
        mask[streamwise_cord[:,0], streamwise_cord[:,1]] = 1
        pivots = np.where(mask.reshape(-1) == 1)
        pivots = np.asarray(pivots).ravel()

        upstream_mask = np.ones((m,n))
        upstream_mask[:origin_x,:] = 0

        sensor_num = len(pivots)

        sensor_params['pivots'] = pivots
        sensor_params['sensor_num'] = sensor_num
        sensor_params['upstream_mask'] = upstream_mask

        dist_from_rear = (origin_x-75)*1/50
        spanwise_total_length = (spanwise_cord[-1,1]-spanwise_cord[0,1])*1/50
        spanwise_separation_length = (spanwise_cord[1,1]-spanwise_cord[0,1])*1/50
        streamwise_total_length = (streamwise_cord[-1,0]-streamwise_cord[0,0])*1/50
        streamwise_separation_length = (streamwise_cord[1,0]-streamwise_cord[0,0])*1/50

        print(f'''Physically:
        Distance from obstacle rear: {dist_from_rear} unit lengths
        Total sensor count: {sensor_num}
        Spanwise:
            Total line length: {spanwise_total_length} unit lengths
            Sensor separation: {spanwise_separation_length} unit lengths
        Streamwise:
            Total line length: {streamwise_total_length} unit lengths
            Sensor separation: {streamwise_separation_length} unit lengths
        ''')

    if x.ndim == 2:
        sensors = x[:, pivots].reshape(n_snapshots, sensor_num)
    else:
        sensors = x[:, :, pivots].reshape(n_snapshots, sensor_num)

    return sensors, sensor_params


def _set_sensor_patch(x, sensor_params):
    if x.ndim == 2:
        n_snapshots, n_pix = x.shape
    elif x.ndim == 3:
        n_snapshots, _ ,n_pix = x.shape

    sensor_num = sensor_params.get('sensor_num')
    pivots = sensor_params.get('pivots')

    if pivots is None:
        m = sensor_params.get('m', 384)
        n = sensor_params.get('n', 199)
        line_start_streamwise = sensor_params.get('line_start_streamwise', 75)
        line_end_streamwise = sensor_params.get('line_end_streamwise', m-1)
        line_start_spanwise = sensor_params.get('line_start_spanwise', 0)
        line_end_spanwise = sensor_params.get('line_end_spanwise', n-1)
        sensor_num_spanwise = sensor_params.get('sensor_num_spanwise', 3)
        sensor_num_streamwise = sensor_params.get('sensor_num_streamwise', 3)

        mask = np.zeros((m,n))
        y_points = np.linspace(line_start_spanwise, line_end_spanwise, num=sensor_num_spanwise, dtype=int)
        x_points = np.linspace(line_start_streamwise, line_end_streamwise, num=sensor_num_streamwise, dtype=int)
        x_cord, y_cord = np.meshgrid(x_points, y_points)
        cords = np.vstack((x_cord.ravel(),y_cord.ravel())).T
        cords = np.unique(cords, axis=0)
        idx = cords[:,0] > 0
        cords = cords[idx,:]

        # if equispaced:
        #     idx = np.linspace(0, x_cord.shape[0]-1, num=sensor_num, dtype=int)
        # else:
        #     random_seed = sensor_params.get('random_seed', 12345)
        #     np.random.seed(random_seed)
        #     idx = np.random.choice(range(x_cord.shape[0]), sensor_num, False)
        
        # x_cord = np.int64(x_cord[idx,:])
        cords = np.int64(cords)

        mask[cords[:,0], cords[:,1]] = 1
        pivots = np.where(mask.reshape(-1) == 1)
        pivots = np.asarray(pivots).ravel()

        sensor_num = len(cords)

        sensor_params['pivots'] = pivots
        sensor_params['sensor_num'] = sensor_num
        
        dist_from_rear = (line_start_streamwise-75)*1/50
        streamwise_total_length = (x_cord[0,-1]-x_cord[0,0])*1/50
        streamwise_separation_length = (x_cord[0,1]-x_cord[0,0])*1/50
        spanwise_total_length = (y_cord[-1,0]-y_cord[0,0])*1/50
        spanwise_separation_length = (y_cord[1,0]-y_cord[0,0])*1/50

        print(f'''Physically:
        Distance from obstacle rear: {dist_from_rear} unit lengths
        Total sensor count: {sensor_num}
        Spanwise:
            Total line length: {spanwise_total_length} unit lengths
            Sensor separation: {spanwise_separation_length} unit lengths
        Streamwise:
            Total line length: {streamwise_total_length} unit lengths
            Sensor separation: {streamwise_separation_length} unit lengths
        ''')

    if x.ndim == 2:
        sensors = x[:, pivots].reshape(n_snapshots, sensor_num)
    else:
        sensors = x[:, :, pivots].reshape(n_snapshots, sensor_num)

    return sensors, sensor_params


def _set_sensor_resolution_reduction(x, sensor_params):
    if x.ndim == 2:
        n_snapshots, n_pix = x.shape
    elif x.ndim == 3:
        n_snapshots, _ ,n_pix = x.shape

    sensor_num = sensor_params.get('sensor_num')
    pivots = sensor_params.get('pivots')

    if pivots is None:
        m = sensor_params.get('m', 384)
        n = sensor_params.get('n', 199)
        line_start_streamwise = sensor_params.get('line_start_streamwise', 75)
        reduction_factor = sensor_params.get('reduction_factor', 1)

        mask = np.zeros((m,n))

        # y_points = np.linspace(line_start_spanwise, line_end_spanwise, num=sensor_num_spanwise, dtype=int)
        # x_points = np.linspace(line_start_streamwise, line_end_streamwise, num=sensor_num_streamwise, dtype=int)
        x_points = np.arange(line_start_streamwise, m, reduction_factor)
        y_points = np.arange(0, n, reduction_factor)
        print(y_points)
        print(x_points)

        x_cord, y_cord = np.meshgrid(x_points, y_points)
        cords = np.vstack((x_cord.ravel(),y_cord.ravel())).T
        cords = np.unique(cords, axis=0)
        idx = cords[:,0] > 0
        cords = cords[idx,:]
        cords = np.int64(cords)

        mask[cords[:,0], cords[:,1]] = 1
        pivots = np.where(mask.reshape(-1) == 1)
        pivots = np.asarray(pivots).ravel()

        sensor_num = len(cords)

        sensor_params['m_red'] = len(x_points)
        sensor_params['n_red'] = len(y_points)
        sensor_params['pivots'] = pivots
        sensor_params['sensor_num'] = sensor_num
        
        dist_from_rear = (line_start_streamwise-75)*1/50
        streamwise_total_length = (x_cord[0,-1]-x_cord[0,0])*1/50
        streamwise_separation_length = (x_cord[0,1]-x_cord[0,0])*1/50
        spanwise_total_length = (y_cord[-1,0]-y_cord[0,0])*1/50
        spanwise_separation_length = (y_cord[1,0]-y_cord[0,0])*1/50

        print(f'''Physically:
        Distance from obstacle rear: {dist_from_rear} unit lengths
        Total sensor count: {sensor_num}
        Spanwise:
            Total line length: {spanwise_total_length} unit lengths
            Sensor separation: {spanwise_separation_length} unit lengths
        Streamwise:
            Total line length: {streamwise_total_length} unit lengths
            Sensor separation: {streamwise_separation_length} unit lengths
        ''')

    if x.ndim == 2:
        sensors = x[:, pivots].reshape(n_snapshots, sensor_num)
    else:
        sensors = x[:, :, pivots].reshape(n_snapshots, sensor_num)

    return sensors, sensor_params


def visualise_extracted_sensor_locations(y, sensor_params, save_folder_path, samples=[0], obstacle=None):
    m = sensor_params.get('m')
    n = sensor_params.get('n')
    sensor_locations = sensor_params.get('pivots')

    xgrid = np.arange(0, m, 1)
    ygrid = np.arange(0, n, 1)
    mX, mY = np.meshgrid(xgrid, ygrid)

    fig, axs = plt.subplots(len(samples), 1, facecolor='white', edgecolor='k', figsize=(7.9, 4.7*len(samples)))
    for idx, sample in enumerate(samples):
        # setting background
        y_sample = y[sample, :, :]

        minmax = np.nanmax(np.abs(y_sample)) * 0.65

        if len(samples) > 1:
            ax = axs[idx]
        else:
            ax = axs
        
        ax.imshow(y_sample.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
        ax.contourf(mX, mY, y_sample.T, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)

        ygrid = range(n)
        xgrid = range(m)
        yv, xv = np.meshgrid(ygrid, xgrid)

        x_sensors = xv.reshape(1, m*n)[:, sensor_locations]
        y_sensors = yv.reshape(1, m*n)[:, sensor_locations]

        ax.scatter(x_sensors, y_sensors, marker='.', color='#ff7f00', s=100, zorder=5)

        for i in range(len(sensor_locations)):
            ax.annotate(f' {i}', (x_sensors[0][i], y_sensors[0][i]), color='#ff7f00')
        ax.set_title('Truth with sensor locations')
        ax.grid(False)
        ax.axis('off')

    plt.tight_layout()
    if obstacle is not None:
        plt.savefig(f'{save_folder_path}/{obstacle}/{obstacle}_sensor_location_visualisation.png', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(f'{save_folder_path}/sensor_location_visualisation.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()


def visualise_extracted_sensor_outputs(sensors, save_folder_path, obstacle=None):
    sensor_num = sensors.shape[-1]
    rows = math.ceil(sensor_num / 5)
    cols = sensor_num % 5 if sensor_num < 5 else 5

    fig, axs = plt.subplots(rows, cols, facecolor='white', edgecolor='k', figsize=(5*cols, 5*rows))
    for row in range(rows):
        for col in range(cols):
            if rows == 1 and cols == 1:
                ax = axs
            elif rows == 1:
                ax = axs[col]
            else:
                ax = axs[row][col]

            idx = row*5+col
            if idx == sensor_num:
                break
            
            ax.plot(sensors[:, :, idx])
            ax.set_title(f'Sensor {idx}')
            
    plt.tight_layout()
    if obstacle is not None:
        plt.savefig(f'{save_folder_path}/{obstacle}/{obstacle}_sensor_output_visualisation.png', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(f'{save_folder_path}/sensor_output_visualisation.png', facecolor=fig.get_facecolor(), edgecolor='none')
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


def visualise_upstream_mask(upstream_mask, y, sensor_params, save_folder_path):
    y_sample = y[0,:,:]
    minmax = np.nanmax(np.abs(y_sample)) * 0.65
    fig, axs = plt.subplots(3, 1, figsize=(8,14), facecolor='white', edgecolor='k')
    axs[0].imshow(upstream_mask.T, cmap='gray')
    axs[0].set_title('upstream binary mask')
    axs[1].imshow(y_sample.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    axs[1].set_title('actual sample')
    axs[2].imshow((y_sample*upstream_mask).T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    axs[2].set_title('masked sample')
    plt.tight_layout()
    plt.savefig(f'{save_folder_path}/upstream_mask.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()


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


def visualise_max_error_comparison(actual, prediction, error, l2_error, save_folder_path, samples=[0], obstacle=None):
    max_error_sample = np.argmax(l2_error)

    actual_sample = actual[max_error_sample,:,:]
    prediction_sample = prediction[max_error_sample,:,:]
    error_sample = error[max_error_sample, :, :]
    l2_error_sample = l2_error[max_error_sample,0]

    print(f'Plotting comparison plot for max L2 error sample, sample {max_error_sample}, L2 error {l2_error_sample*100:.4f}%')
    # initialising plots
    _, m, n = actual.shape

    x = np.arange(0, m, 1)
    y = np.arange(0, n, 1)
    mX, mY = np.meshgrid(x, y)
    minmax = np.nanmax(np.abs(actual_sample)) * 0.65

    # Plotting
    fig, axs = plt.subplots(3, facecolor="white",  edgecolor='k', figsize=(7.9,16.9))
    fig.suptitle('Output comparison')

    axs[0].imshow(actual_sample.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    axs[0].contourf(mX, mY, actual_sample.T, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    axs[0].set_title('Actual')
    axs[0].axis('off')

    axs[1].imshow(prediction_sample.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    axs[1].contourf(mX, mY, prediction_sample.T, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    axs[1].set_title('Predicted')          
    axs[1].axis('off')

    im2 = axs[2].imshow(error_sample.T, cmap='gray_r', interpolation='none', vmin=0, vmax=error_sample.max())
    axs[2].set_title(f'Absolute difference. Overall L2 error = {l2_error_sample*100:.4f}%')
    axs[2].axis('off')
    fig.colorbar(im2, ax=axs[2], orientation='horizontal')

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