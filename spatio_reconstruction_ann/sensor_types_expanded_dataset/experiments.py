# ---------------------- changed this ----------------------
from sensor_types_expanded_dataset.base import experiment
# ----------------------------------------------------------

import os
import json

def experiment_1(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_1'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['sensor_type'] = 'wall'
    data['sensor_parameters']['origin_x'] = 50
    data['sensor_parameters']['origin_y'] = 100
    data['sensor_parameters']['rad'] = 36
    data['sensor_parameters']['sensor_num'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'random'
    data['sensor_parameters']['random_seed'] = 815

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_2(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_2'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['sensor_type'] = 'wall'
    data['sensor_parameters']['origin_x'] = 50
    data['sensor_parameters']['origin_y'] = 100
    data['sensor_parameters']['rad'] = 36
    data['sensor_parameters']['sensor_num'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'rear_equispace'
    data['sensor_parameters']['random_seed'] = 815

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_3(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_3'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # line, near to obstacle
    data['sensor_parameters']['sensor_type'] = 'line'
    data['sensor_parameters']['line_start'] = 50
    data['sensor_parameters']['line_end'] = 151
    data['sensor_parameters']['origin_x'] = 80
    data['sensor_parameters']['sensor_num'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_4(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_4'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # line
    data['sensor_parameters']['sensor_type'] = 'line'
    data['sensor_parameters']['line_start'] = 50
    data['sensor_parameters']['line_end'] = 151
    data['sensor_parameters']['origin_x'] = 100
    data['sensor_parameters']['sensor_num'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_5(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_5'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # line, narrow
    data['sensor_parameters']['sensor_type'] = 'line'
    data['sensor_parameters']['line_start'] = 80
    data['sensor_parameters']['line_end'] = 121
    data['sensor_parameters']['origin_x'] = 100
    data['sensor_parameters']['sensor_num'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_6(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_6'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # line, wide
    data['sensor_parameters']['sensor_type'] = 'line'
    data['sensor_parameters']['line_start'] = 30
    data['sensor_parameters']['line_end'] = 171
    data['sensor_parameters']['origin_x'] = 100
    data['sensor_parameters']['sensor_num'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)



def experiment_7(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_7'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # line, perpendicular
    data['sensor_parameters']['sensor_type'] = 'line'
    data['sensor_parameters']['spanwise'] = False
    data['sensor_parameters']['line_start'] = 80
    data['sensor_parameters']['line_end'] = 181
    data['sensor_parameters']['origin_y'] = 100
    data['sensor_parameters']['sensor_num'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)



def experiment_8(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_8'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # line, perpendicular, short
    data['sensor_parameters']['sensor_type'] = 'line'
    data['sensor_parameters']['spanwise'] = False
    data['sensor_parameters']['line_start'] = 80
    data['sensor_parameters']['line_end'] = 151
    data['sensor_parameters']['origin_y'] = 100
    data['sensor_parameters']['sensor_num'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_9(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_9'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # line, perpendicular, long
    data['sensor_parameters']['sensor_type'] = 'line'
    data['sensor_parameters']['spanwise'] = False
    data['sensor_parameters']['line_start'] = 80
    data['sensor_parameters']['line_end'] = 211
    data['sensor_parameters']['origin_y'] = 100
    data['sensor_parameters']['sensor_num'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_10(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_10'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # T
    data['sensor_parameters']['sensor_type'] = 't'
    data['sensor_parameters']['line_start_spanwise'] = 50
    data['sensor_parameters']['line_end_spanwise'] = 151
    data['sensor_parameters']['line_end_streamwise'] = 181
    data['sensor_parameters']['origin_x'] = 80
    data['sensor_parameters']['origin_y'] = 100
    data['sensor_parameters']['sensor_num_spanwise'] = 3
    data['sensor_parameters']['sensor_num_streamwise'] = 3
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_11(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_11'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # T, dense
    data['sensor_parameters']['sensor_type'] = 't'
    data['sensor_parameters']['line_start_spanwise'] = 50
    data['sensor_parameters']['line_end_spanwise'] = 151
    data['sensor_parameters']['line_end_streamwise'] = 181
    data['sensor_parameters']['origin_x'] = 80
    data['sensor_parameters']['origin_y'] = 100
    data['sensor_parameters']['sensor_num_spanwise'] = 5
    data['sensor_parameters']['sensor_num_streamwise'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_12(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_12'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # T, dense perpendicular line
    data['sensor_parameters']['sensor_type'] = 't'
    data['sensor_parameters']['line_start_spanwise'] = 50
    data['sensor_parameters']['line_end_spanwise'] = 151
    data['sensor_parameters']['line_end_streamwise'] = 181
    data['sensor_parameters']['origin_x'] = 80
    data['sensor_parameters']['origin_y'] = 100
    data['sensor_parameters']['sensor_num_spanwise'] = 3
    data['sensor_parameters']['sensor_num_streamwise'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_13(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_13'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # T, dense line
    data['sensor_parameters']['sensor_type'] = 't'
    data['sensor_parameters']['line_start_spanwise'] = 50
    data['sensor_parameters']['line_end_spanwise'] = 151
    data['sensor_parameters']['line_end_streamwise'] = 181
    data['sensor_parameters']['origin_x'] = 80
    data['sensor_parameters']['origin_y'] = 100
    data['sensor_parameters']['sensor_num_spanwise'] = 5
    data['sensor_parameters']['sensor_num_streamwise'] = 3
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_14(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_14'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # patch
    data['sensor_parameters']['sensor_type'] = 'patch'
    data['sensor_parameters']['line_start_streamwise'] = 80
    data['sensor_parameters']['line_end_streamwise'] = 180
    data['sensor_parameters']['line_start_spanwise'] = 50
    data['sensor_parameters']['line_end_spanwise'] = 150
    data['sensor_parameters']['sensor_num_spanwise'] = 3
    data['sensor_parameters']['sensor_num_streamwise'] = 3
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_15(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_15'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # patch, dense
    data['sensor_parameters']['sensor_type'] = 'patch'
    data['sensor_parameters']['line_start_streamwise'] = 80
    data['sensor_parameters']['line_end_streamwise'] = 180
    data['sensor_parameters']['line_start_spanwise'] = 50
    data['sensor_parameters']['line_end_spanwise'] = 150
    data['sensor_parameters']['sensor_num_spanwise'] = 4
    data['sensor_parameters']['sensor_num_streamwise'] = 4
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)

def experiment_16(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_16'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # patch, sr cnn equivalent
    data['sensor_parameters']['sensor_type'] = 'patch'
    data['sensor_parameters']['line_start_streamwise'] = 80
    data['sensor_parameters']['line_end_streamwise'] = 208
    data['sensor_parameters']['line_start_spanwise'] = 36
    data['sensor_parameters']['line_end_spanwise'] = 164
    data['sensor_parameters']['sensor_num_spanwise'] = 16
    data['sensor_parameters']['sensor_num_streamwise'] = 16
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_17(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_17'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # wall, very dense
    data['sensor_parameters']['sensor_type'] = 'wall'
    data['sensor_parameters']['origin_x'] = 50
    data['sensor_parameters']['origin_y'] = 100
    data['sensor_parameters']['rad'] = 36
    data['sensor_parameters']['sensor_num'] = 15
    data['sensor_parameters']['sensor_distribution'] = 'rear_equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_18(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_18'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # line, very dense
    data['sensor_parameters']['sensor_type'] = 'line'
    data['sensor_parameters']['origin_x'] = 100
    data['sensor_parameters']['sensor_num'] = 15
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_19(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_19'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    # patch, whole domain
    data['sensor_parameters']['sensor_type'] = 'patch'
    data['sensor_parameters']['line_start_streamwise'] = 80
    data['sensor_parameters']['sensor_num_spanwise'] = 3
    data['sensor_parameters']['sensor_num_streamwise'] = 5
    data['sensor_parameters']['sensor_distribution'] = 'equispace'

    data['training_parameters']['early_stopping'] = True
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)



# ---------------------- runner functions ----------------------
def get_save_folder(save_folder_name):
    cwd = os.path.dirname(os.path.realpath(__file__))   

    save_folder_path = os.path.join(cwd, save_folder_name)
    if not os.path.isdir(save_folder_path):
        os.mkdir(save_folder_path)
        print(f'created folder: {save_folder_path}')

    return save_folder_path

def get_config():
    cwd = os.path.dirname(os.path.realpath(__file__))  

    json_file_path = os.path.join(cwd, 'config_template.json')
    with open(json_file_path) as json_file:
        data = json.load(json_file)

    return data

def save_config(data, save_folder_name):
    cwd = os.path.dirname(os.path.realpath(__file__))

    json_save_file_path = os.path.join(cwd, save_folder_name, 'config.json')
    with open(json_save_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        print(f'config saved at: {save_folder_name}')

def save_output(output, save_folder_name):
    cwd = os.path.dirname(os.path.realpath(__file__))

    json_output_file_path = os.path.join(cwd, save_folder_name, 'output.json')
    if not os.path.isfile(json_output_file_path):
        with open(json_output_file_path, 'w') as json_file:
            json.dump({'counter': 0, 0: output}, json_file, indent=4)
            print(f'output saved at: {save_folder_name}')
    else:
        with open(json_output_file_path) as json_file:
            previous_output = json.load(json_file)
            previous_counter = previous_output['counter']
            current_counter = previous_counter + 1
            previous_output['counter'] = current_counter
            previous_output[current_counter] = output
            with open(json_output_file_path, 'w') as json_file:
                json.dump(previous_output, json_file, indent=4)
                print(f'output updated at: {save_folder_name}')