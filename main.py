# main script
"""
@author: mengxue.Zhang
"""

from model import get_model
from train_test_scripts import train_model, test_noise_model

# 'DEBUG' mode print the training processing
mode = 'DEBUG'
# mode = 0
run_times = 20


def soc_experiment(base_dir='./soe_no_aug'):
    train_prefix = 'train'
    test_prefix = 'test'
    model = get_model(classes=10)
    train_model(model, False, base_dir, train_prefix, test_prefix, False, run_times)


def soc_aug_experiment(base_dir='./soe', train_shapes=[91, 91]):
    train_prefix = 'train'
    test_prefix = 'test'
    model = get_model(classes=10)
    train_model(model, mode, base_dir, train_prefix, test_prefix, False, run_times, train_shapes=train_shapes)


def soc_reduce_sample_experiement(base_dir='./reduce', train_shapes=[88, 88]):
    name_str = ['p90', 'p80', 'p70', 'p60', 'p50', 'p40', 'p30', 'p20', 'p10']

    for name in name_str:
        train_prefix = 'train_' + name
        b_dir = base_dir + '/' + name
        test_prefix = 'test'
        model = get_model(classes=10)
        train_model(model, mode, b_dir, train_prefix, test_prefix, True, 10, train_shapes=train_shapes)


def soc_noise(model_path='./soe_no_aug', base_dir='./noise', ratios=[0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.35, 0.40, 0.45, 0.50]):
    # noise experiment use the no augment model.
    # soc_experiment(model_path)

    model = get_model(classes=10)
    test_noise_model(model, model_path, model_num=run_times, mode='', base_dir=base_dir, ratios=ratios)
    print('results store in the ' + base_dir)


def eoc_vv_experiment(base_dir='./vv', train_shapes=[91, 91], save_best=1):
    # In the vv experiment, we only record the best valid acc.
    train_prefix = 'train_eoc2'
    test_prefix = 'test_eoc2_vv'
    model = get_model(classes=4)
    train_model(model, False, base_dir, train_prefix, test_prefix, False, run_times, train_shapes=train_shapes, save_best=save_best)


def eoc_cv_experiment( base_dir='./cv', train_shapes=[89, 89], save_best=1):
    # In the cv experiment, we only record the best valid acc.
    train_prefix = 'train_eoc2'
    test_prefix = 'test_eoc2_cv'
    model = get_model(classes=4)
    train_model(model, False, base_dir, train_prefix, test_prefix, False, run_times, train_shapes=train_shapes, save_best=save_best)



soc_aug_experiment(base_dir='./soe', train_shapes=[91, 91])

