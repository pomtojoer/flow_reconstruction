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
    choices=[0, 1, 2, 3, 4]
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
            # No scaling
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # Mean centring
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue
        
elif args.parent == 1:
    # Resolution reduction, cropped domain, DSC/MS, cylinder
    if child_exps is None:
        child_exps = [1,2,3,4,5,6]
        child_exps = child_exps[::-1]
    from resolution_reduction import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # 1/1 resolution reduction
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # 1/2 resolution reduction
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # 1/4 resolution reduction
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # 1/8 resolution reduction
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            # 1/16 resolution reduction
            exp.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            # 1/32 resolution reduction
            exp.experiment_6(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

elif args.parent == 2:
    # Resolution reduction, full domain, DSC/MS, cylinder
    if child_exps is None:
        child_exps = [1,2,3,4,5,6]
    from resolution_reduction_expanded_domain import experiments as exp3
    for child_exp in child_exps:
        if child_exp == 1:
            # 1/1 resolution reduction
            exp3.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # 1/2 resolution reduction
            exp3.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # 1/4 resolution reduction
            exp3.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # 1/8 resolution reduction
            exp3.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            # 1/16 resolution reduction
            exp3.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            # 1/32 resolution reduction
            exp3.experiment_6(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

elif args.parent == 3:
    # Resolution reduction, full domain, DSC/MS, primitives
    if child_exps is None:
        child_exps = [1,2,3,4,5,6]
    from resolution_reduction_expanded_dataset import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # 1/1 resolution reduction
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # 1/2 resolution reduction
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # 1/4 resolution reduction
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # 1/8 resolution reduction
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            # 1/16 resolution reduction
            exp.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            # 1/32 resolution reduction
            exp.experiment_6(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


elif args.parent == 4:
    # Model comparison
    if child_exps is None:
        child_exps = [1,2,3,4,5,6,7,8]
        print(child_exps)
    from model_comparison import experiments as exp
    for child_exp in child_exps:
        # Model comparison
        if child_exp == 1:
            # interpolation: bicubic
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # interpolation: bilinear
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # interpolation: lanczos3
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # interpolation: lanczos5
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            # interpolation: gaussian
            exp.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            # interpolation: nearest
            exp.experiment_6(train_model=train_model)
            K.clear_session()
        elif child_exp == 7:
            # CNN: SRCNN
            exp.experiment_7(train_model=train_model)
            K.clear_session()
        elif child_exp == 8:
            # CNN: SCNN
            exp.experiment_8(train_model=train_model)
            K.clear_session()
        elif child_exp == 9:
            # CNN: DSC/MS
            exp.experiment_9(train_model=train_model)
            K.clear_session()

        # Model modification
        elif child_exp == 10:
            # DSC/MS: bicubic preprocessing layer
            exp.experiment_10(train_model=train_model)
            K.clear_session()
        elif child_exp == 11:
            # SCNN: VGG modification
            exp.experiment_11(train_model=train_model)
            K.clear_session()
        elif child_exp == 12:
            # DSC/MS: VGG modification
            exp.experiment_12(train_model=train_model)
            K.clear_session()
        elif child_exp == 13:
            # SCNN: skip connection
            exp.experiment_13(train_model=train_model)
            K.clear_session()
        elif child_exp == 14:
            # SCNN: custom loss function (sum squared error)
            exp.experiment_14(train_model=train_model)
            K.clear_session()
        elif child_exp == 15:
            # DSC/MS: custom loss function (sum squared error)
            exp.experiment_15(train_model=train_model)
            K.clear_session()

        # SCNN accuracy and expanded dataset
        elif child_exp == 16:
            # SCNN: 1/32 resolution reduction, cylinder
            exp.experiment_16(train_model=train_model)
            K.clear_session()
        elif child_exp == 17:
            # SCNN: 1/32 resolution reduction, primitives
            exp.experiment_17(train_model=train_model)
            K.clear_session()
        elif child_exp == 18:
            # SCNN: 1/8 resolution reduction, primitives
            exp.experiment_18(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

else:
    print(f'unrecognised parent {args.parent}')

K.clear_session()

