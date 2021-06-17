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
    if child_exps is None:
        child_exps = [1,2,3]
    from data_scaling import experiments as exp1
    for child_exp in child_exps:
        if child_exp == 1:
            exp1.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            exp1.experiment_2(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue
        
elif args.parent == 1:
    if child_exps is None:
        child_exps = [1,2,3,4,5,6]
        child_exps = child_exps[::-1]
        print(child_exps)
    from resolution_reduction import experiments as exp2
    for child_exp in child_exps:
        if child_exp == 1:
            exp2.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            exp2.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            exp2.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            exp2.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            exp2.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            exp2.experiment_6(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

elif args.parent == 2:
    if child_exps is None:
        child_exps = [1,2,3,4,5,6]
    from resolution_reduction_expanded_domain import experiments as exp3
    for child_exp in child_exps:
        if child_exp == 1:
            exp3.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            exp3.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            exp3.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            exp3.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            exp3.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            exp3.experiment_6(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

elif args.parent == 3:
    if child_exps is None:
        child_exps = [1,2,3,4,5,6]
    from resolution_reduction_expanded_dataset import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            exp.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            exp.experiment_6(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue


elif args.parent == 4:
    if child_exps is None:
        child_exps = [1,2,3,4,5,6,7,8]
        print(child_exps)
    from model_comparison import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        elif child_exp == 5:
            exp.experiment_5(train_model=train_model)
            K.clear_session()
        elif child_exp == 6:
            exp.experiment_6(train_model=train_model)
            K.clear_session()
        elif child_exp == 7:
            exp.experiment_7(train_model=train_model)
            K.clear_session()
        elif child_exp == 8:
            exp.experiment_8(train_model=train_model)
            K.clear_session()
        elif child_exp == 9:
            exp.experiment_9(train_model=train_model)
            K.clear_session()
        elif child_exp == 10:
            exp.experiment_10(train_model=train_model)
            K.clear_session()
        elif child_exp == 11:
            exp.experiment_11(train_model=train_model)
            K.clear_session()
        elif child_exp == 12:
            exp.experiment_12(train_model=train_model)
            K.clear_session()
        elif child_exp == 13:
            exp.experiment_13(train_model=train_model)
            K.clear_session()
        elif child_exp == 14:
            exp.experiment_14(train_model=train_model)
            K.clear_session()
        elif child_exp == 15:
            exp.experiment_15(train_model=train_model)
            K.clear_session()
        elif child_exp == 16:
            exp.experiment_16(train_model=train_model)
            K.clear_session()
        elif child_exp == 17:
            exp.experiment_17(train_model=train_model)
            K.clear_session()
        elif child_exp == 18:
            exp.experiment_18(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue

else:
    print(f'unrecognised parent {args.parent}')

K.clear_session()

