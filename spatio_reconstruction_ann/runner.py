import argparse

# parsing arguments
parser = argparse.ArgumentParser(
    prog='runner',
    description='Runner to run individual experiments'
)

parser.add_argument(
    'parent',
    help='Parent experiment. 0 = data scaling, 1 = sensor placement',
    type=int,
    choices=[0, 1, 2]
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

if args.parent == 0:
    # Data scaling
    if child_exps is None:
        child_exps = [1,2,3]
    from data_scaling import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # Mean centring
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # Standardisation
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # Normalisation
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # Unscaled
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue
        
elif args.parent == 1:
    # Sensor placement
    if child_exps is None:
        child_exps = [1,2,3]
    from sensor_types import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # Wall, manual placement, 5
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # Wall, equispaced placement, 5
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # Line, spanwise, 5
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # Line, spanwise, 5, offset
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            # Line, spanwise, 5, offset, narrow
            exp.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            # Line, spanwise, 5, offset, wide
            exp.experiment_6(train_model=train_model)
            K.clear_session()
        elif child_exp == 7:
            # Line, streamwise, 5
            exp.experiment_7(train_model=train_model)
            K.clear_session()
        elif child_exp == 8:
            # Line, streamwise, 5, narrow
            exp.experiment_8(train_model=train_model)
            K.clear_session()
        elif child_exp == 9:
            # Line, streamwise, 5, wide
            exp.experiment_9(train_model=train_model)
            K.clear_session()
        elif child_exp == 10:
            # T, 5
            exp.experiment_10(train_model=train_model)
            K.clear_session()
        elif child_exp == 11:
            # T, 9, dense
            exp.experiment_11(train_model=train_model)
            K.clear_session()
        elif child_exp == 12:
            # T, 7, streamwise dense
            exp.experiment_12(train_model=train_model)
            K.clear_session()
        elif child_exp == 13:
            # T, 7, spanwise dense
            exp.experiment_13(train_model=train_model)
            K.clear_session()
        elif child_exp == 14:
            # Patch, 3x3
            exp.experiment_14(train_model=train_model)
            K.clear_session()
        elif child_exp == 15:
            # Patch, 4x4
            exp.experiment_15(train_model=train_model)
            K.clear_session()
        elif child_exp == 16:
            # Patch, 16x16
            exp.experiment_16(train_model=train_model)
            K.clear_session()
        elif child_exp == 17:
            # Wall, equispaced placement, 15, dense
            exp.experiment_17(train_model=train_model)
            K.clear_session()
        elif child_exp == 18:
            # Line, spanwise, 15, dense
            exp.experiment_18(train_model=train_model)
            K.clear_session()
        elif child_exp == 19:
            # Patch, full domain, 3x5
            exp.experiment_19(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

elif args.parent == 2:
    # Sensor placement, expanded dataset
    if child_exps is None:
        child_exps = [1,2,3]
    from sensor_types_expanded_dataset import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # Wall, manual placement, 5
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # Wall, equispaced placement, 5
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # Line, spanwise, 5
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # Line, spanwise, 5, offset
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            # Line, spanwise, 5, offset, narrow
            exp.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            # Line, spanwise, 5, offset, wide
            exp.experiment_6(train_model=train_model)
            K.clear_session()
        elif child_exp == 7:
            # Line, streamwise, 5
            exp.experiment_7(train_model=train_model)
            K.clear_session()
        elif child_exp == 8:
            # Line, streamwise, 5, narrow
            exp.experiment_8(train_model=train_model)
            K.clear_session()
        elif child_exp == 9:
            # Line, streamwise, 5, wide
            exp.experiment_9(train_model=train_model)
            K.clear_session()
        elif child_exp == 10:
            # T, 5
            exp.experiment_10(train_model=train_model)
            K.clear_session()
        elif child_exp == 11:
            # T, 9, dense
            exp.experiment_11(train_model=train_model)
            K.clear_session()
        elif child_exp == 12:
            # T, 7, streamwise dense
            exp.experiment_12(train_model=train_model)
            K.clear_session()
        elif child_exp == 13:
            # T, 7, spanwise dense
            exp.experiment_13(train_model=train_model)
            K.clear_session()
        elif child_exp == 14:
            # Patch, 3x3
            exp.experiment_14(train_model=train_model)
            K.clear_session()
        elif child_exp == 15:
            # Patch, 4x4
            exp.experiment_15(train_model=train_model)
            K.clear_session()
        elif child_exp == 16:
            # Patch, 16x16
            exp.experiment_16(train_model=train_model)
            K.clear_session()
        elif child_exp == 17:
            # Wall, equispaced placement, 15, dense
            exp.experiment_17(train_model=train_model)
            K.clear_session()
        elif child_exp == 18:
            # Line, spanwise, 15, dense
            exp.experiment_18(train_model=train_model)
            K.clear_session()
        elif child_exp == 19:
            # Patch, full domain, 3x5
            exp.experiment_19(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

else:
    print(f'unrecognised parent {args.parent}')


K.clear_session()