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
    choices=[0,]
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
    # Dataset augmentation
    if child_exps is None:
        child_exps = [1,2,3,4]
    from dataset_augmentation import experiments as exp
    for child_exp in child_exps:
        if child_exp == 1:
            # Original dataset 
            exp.experiment_1(train_model=train_model)
            K.clear_session()
        elif child_exp == 2:
            # Bezier 200 dataset
            exp.experiment_2(train_model=train_model)
            K.clear_session()
        elif child_exp == 3:
            # Bezier 200 dataset flipped
            exp.experiment_3(train_model=train_model)
            K.clear_session()
        elif child_exp == 4:
            # Bezier 200 dataset flipped with SDF
            exp.experiment_4(train_model=train_model)
            K.clear_session()
        else:
            print(f'unrecognised child {child_exp}, skipping')
            continue
        

else:
    print(f'unrecognised parent {args.parent}')

# from data_scaling import experiments as exp1

# from tensorflow.keras import backend as K

# K.clear_session()

# exp1.experiment_1(train_model=True)
# K.clear_session()
# exp1.experiment_2(train_model=True)
# K.clear_session()
# exp1.experiment_3(train_model=True)
# K.clear_session()