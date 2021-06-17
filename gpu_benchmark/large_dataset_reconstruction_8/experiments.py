# ---------------------- changed this ----------------------
from large_dataset_reconstruction_8.base import experiment
# ----------------------------------------------------------

import os
import json

def experiment_1(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_1_t4'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_2(train_model=False):
    # P3 instance
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_2_v100'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_3(train_model=False):
    # P4 instance
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_3_a100'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
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