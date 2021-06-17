# ---------------------- changed this ----------------------
# ----------------------------------------------------------

import os
import json

def experiment_1(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.interpolation import experiment
    save_folder_name = 'experiment_1'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'interpolation'
    data['model_parameters']['model_method'] = 'bicubic'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)

def experiment_2(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.interpolation import experiment
    save_folder_name = 'experiment_2'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'interpolation'
    data['model_parameters']['model_method'] = 'bilinear'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)

def experiment_3(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.interpolation import experiment
    save_folder_name = 'experiment_3'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'interpolation'
    data['model_parameters']['model_method'] = 'lanczos3'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)

def experiment_4(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.interpolation import experiment
    save_folder_name = 'experiment_4'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'interpolation'
    data['model_parameters']['model_method'] = 'lanczos5'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)

def experiment_5(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.interpolation import experiment
    save_folder_name = 'experiment_5'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'interpolation'
    data['model_parameters']['model_method'] = 'gaussian'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)

def experiment_6(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.interpolation import experiment
    save_folder_name = 'experiment_6'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'interpolation'
    data['model_parameters']['model_method'] = 'nearest'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_7(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_7'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'srcnn'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_8(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_8'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'scnn'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_9(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_9'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'equal'
    data['model_parameters']['model_name'] = 'dsc_ms'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_10(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_10'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'dsc_ms_mod'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_11(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_11'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'scnn_vgg_mod'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_12(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_12'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'dsc_ms_vgg_mod'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_13(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_13'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'scnn_sc_mod'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_14(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_14'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'scnn_custom_loss'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_15(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_15'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'dsc_ms_mod_custom_loss'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_16(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_16'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['dataset_parameters']['dataset_filepath'] = "primitives/cylinder/w_z.npy"

    data['sensor_parameters']['down_res'] = 32
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'scnn'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)


def experiment_17(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_17'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['dataset_parameters']['dataset_filepath'] = "primitives/combined_primitives/w_z.npy"
    data['dataset_parameters']['oscillation_start_idx'] = 0

    data['sensor_parameters']['down_res'] = 32
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'scnn'
    # ---------------------------------------------------------------

    save_config(data, save_folder_name)
    output = experiment(save_folder_path, data, train_model)
    save_output(output, save_folder_name)

def experiment_18(train_model=False):
    # ---------------------- changed this ----------------------
    from model_comparison.tf_base import experiment
    save_folder_name = 'experiment_18'
    # ----------------------------------------------------------

    save_folder_path = get_save_folder(save_folder_name)
    data = get_config()
    
    # ---------------------- changed parameter ----------------------
    data['dataset_parameters']['dataset_filepath'] = "primitives/combined_primitives/w_z.npy"
    data['dataset_parameters']['oscillation_start_idx'] = 0

    data['sensor_parameters']['down_res'] = 8
    data['sensor_parameters']['sensor_type'] = 'reduce'
    data['model_parameters']['model_name'] = 'scnn'
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