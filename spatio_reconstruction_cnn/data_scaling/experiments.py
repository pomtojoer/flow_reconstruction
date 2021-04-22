from data_scaling.base import experiment

import os
import json

def experiment_1(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_1'
    # ----------------------------------------------------------

    cwd = os.path.dirname(os.path.realpath(__file__))

    save_folder_path = os.path.join(cwd, save_folder_name)
    if not os.path.isdir(save_folder_path):
        os.mkdir(save_folder_path)
        print(f'created folder: {save_folder_path}')
    
    json_file_path = os.path.join(cwd, 'config_template.json')
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    
    # ---------------------- changed parameter ----------------------
    data['data_scaling_parameters']['scaling_method'] = 'none'
    # ---------------------------------------------------------------

    json_save_file_path = os.path.join(cwd, save_folder_name, 'config.json')
    with open(json_save_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        print(f'config saved at: {save_folder_name}')

    output = experiment(save_folder_path, data, train_model)
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


def experiment_2(train_model=False):
    # ---------------------- changed this ----------------------
    save_folder_name = 'experiment_2'
    # ----------------------------------------------------------

    cwd = os.path.dirname(os.path.realpath(__file__))

    save_folder_path = os.path.join(cwd, save_folder_name)
    if not os.path.isdir(save_folder_path):
        os.mkdir(save_folder_path)
        print(f'created folder: {save_folder_path}')
    
    json_file_path = os.path.join(cwd, 'config_template.json')
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    
    # ---------------------- changed parameter ----------------------
    data['data_scaling_parameters']['scaling_method'] = 'center'
    # ---------------------------------------------------------------

    json_save_file_path = os.path.join(cwd, save_folder_name, 'config.json')
    with open(json_save_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        print(f'config saved at: {save_folder_name}')

    output = experiment(save_folder_path, data, train_model)
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
