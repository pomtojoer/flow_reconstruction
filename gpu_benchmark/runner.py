import argparse

# parsing arguments
parser = argparse.ArgumentParser(
    prog='runner',
    description='Runner to run individual experiments'
)

parser.add_argument(
    'parent',
    help='Parent experiment',
    type=int,
    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
)

parser.add_argument(
    '-c',
    '--child',
    help='Child experiment',
    type=int,
    action='append', 
    nargs='+'
)

parser.add_argument(
    '-t',
    '--train',
    help='Train model. 0 = False, 1 = True',
    type=int,
    choices=[0,1]
)

# gpus_experiments = [
#     'multiple_reconstruction_1',
#     'multiple_reconstruction_2',
#     'multiple_reconstruction_4',
#     'multiple_reconstruction_8',
# ]

# experiments = ['experiment_1_t4_1', 'experiment_1_t4_2', 'experiment_1_t4_3', 
#                'experiment_2_v100_1', 'experiment_2_v100_2', 'experiment_2_v100_3',
#                'experiment_3_a100_1', 'experiment_3_a100_2', 'experiment_3_a100_3',]


args = parser.parse_args()

from tensorflow.keras import backend as K
K.clear_session()

if args.child is not None:
    child_exps = [j for i in args.child for j in i]
    child_exps = list(set(child_exps))
    child_exps.sort()
else:
    child_exps = None

if args.train is None:
    train_model = False
else:
    train_model = True if args.train == 1 else False

# 1 GPU
if args.parent == 0:
    # single reconstruction, DSC/MS, 1 GPU
    if child_exps is None:
        child_exps = [1,2,3]
    from reconstruction_1 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # g4dn instance t4
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue
        
elif args.parent == 1:
    # multiple reconstruction, DSC/MS, 1 GPU
    if child_exps is None:
        child_exps = [1,2,3]
    from multiple_reconstruction_1 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # g4dn instance t4
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

elif args.parent == 2:
    # single reconstruction, DSC/MS, CPU
    if child_exps is None:
        child_exps = [1,2,3]
    from reconstruction_cpu import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # T4 xlarge (16 cores)
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100 (32 cores)
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100 (48 cores)
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # T4 metal (96 cores)
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

elif args.parent == 3:
    # large dataset reconstruction, SCNN, 1 GPU, 1/8 resolution reduction
    if child_exps is None:
        child_exps = [1,2,3]
    from large_dataset_reconstruction import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

elif args.parent == 4:
    # single reconstruction, SCNN, 1 GPU
    if child_exps is None:
        child_exps = [1,2,3]
    from reconstruction_scnn_1 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # g4dn instance t4
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


# multi 8 GPU
elif args.parent == 5:
    # single reconstruction, DSC/MS, 8 GPU
    if child_exps is None:
        child_exps = [1,2,3]
    from reconstruction_8 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # g4dn instance t4
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

elif args.parent == 6:
    # multiple reconstruction, DSC/MS, 8 GPU
    if child_exps is None:
        child_exps = [1,2,3]
    from multiple_reconstruction_8 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # g4dn instance t4
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


elif args.parent == 7:
    # large dataset reconstruction, SCNN, 8 GPU, 1/8 resolution reduction
    if child_exps is None:
        child_exps = [1,2,3]
    from large_dataset_reconstruction_8 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # g4dn instance t4
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


elif args.parent == 8:
    # single reconstruction, SCNN, 1 GPU
    if child_exps is None:
        child_exps = [1,2,3]
    from reconstruction_scnn_8 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # g4dn instance t4
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


elif args.parent == 9:
    # batch size comparison, g4dn, T4
    if child_exps is None:
        child_exps = [1,2,3,4,5,6,7]
    from batch_size_t4 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # batch size = 32
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # batch size = 64
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # batch size = 128
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # batch size = 256
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            # batch size = 512
            exp.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            # batch size = 1024
            exp.experiment_6(train_model=train_model)
            K.clear_session()
        elif child_exp == 7:
            # batch size = 2048
            exp.experiment_7(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


# multi 2 GPU
elif args.parent == 10:
    # multiple reconstruction, DSC/MS, 2 GPU
    if child_exps is None:
        child_exps = [1,2,3]
    from multiple_reconstruction_2 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # g4dn instance t4
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


# multi 4 GPU
elif args.parent == 11:
    # multiple reconstruction, DSC/MS, 4 GPU
    if child_exps is None:
        child_exps = [1,2,3]
    from multiple_reconstruction_4 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # g4dn instance t4
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


# multi 6 GPU
elif args.parent == 12:
    # multiple reconstruction, DSC/MS, 6 GPU
    if child_exps is None:
        child_exps = [1,2,3]
    from multiple_reconstruction_6 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # g4dn instance t4
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


# multi 8 GPU
elif args.parent == 13:
    # batch size comparison, p4, a100
    if child_exps is None:
        child_exps = [1,2,3,4,5,6,7]
    from batch_size_a100 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # batch size = 32
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # batch size = 64
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # batch size = 128
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # batch size = 256
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            # batch size = 512
            exp.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            # batch size = 1024
            exp.experiment_6(train_model=train_model)
            K.clear_session()
        elif child_exp == 7:
            # batch size = 2048
            exp.experiment_7(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


elif args.parent == 14:
    # large dataset reconstruction, SCNN, 8 GPU, 1/32 resolution reduction
    if child_exps is None:
        child_exps = [1,2,3]
        print(child_exps)
    from large_dataset_reconstruction_8_32 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # P3 instance v100
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # P4 instance a100
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


elif args.parent == 15:
    # batch size comparison, p3, v100
    if child_exps is None:
        child_exps = [1,2,3,4,5,6,7]
        print(child_exps)
    from batch_size_v100 import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # batch size = 32
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2
        # batch size = 64
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # batch size = 128
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # batch size = 256
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            # batch size = 512
            exp.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            # batch size = 1024
            exp.experiment_6(train_model=train_model)
            K.clear_session()
        elif child_exp == 7:
            # batch size = 2048
            exp.experiment_7(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

else:
    print(f'unrecognised parent {args.parent}')

K.clear_session()

