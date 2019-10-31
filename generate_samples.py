# generate train and test samples script
"""
@author: mengxue.Zhang
"""

import os
import numpy as np
import math
import scipy.io as sio


# constant values
train_base_dir = './MSTAR-10/train/'
test_base_dir = './MSTAR-10/test/'
eoc2_base_dir = './MSTAR-10/eoc/'
mat_base_dir = './input'

random_obj = np.random.RandomState(seed=0)
verbose = True
class_name = {'2S1':0, 'BMP2':1, 'BRDM_2':2, 'BTR60':3, 'BTR70':4, 'D7':5, 'T62':6, 'T72':7, 'ZIL131':8, 'ZSU_23_4':9}


def list_dir_or_file(type='dir', base_path='./MSTAR-10/train/', forbid='txt'):
    if not os.path.exists(base_path):
        print('base_path is not exist calling list_dir_or_file!')
        return []
    list_path = os.listdir(base_path)
    list_result = []
    for item in list_path:
        child = os.path.join(base_path, item)
        if type == 'dir':
            if os.path.isdir(child):
                list_result.append(child)
        elif type == 'file':
            if child.find(forbid)==-1:
                if os.path.isfile(child):
                    list_result.append(child)
        else:
            print('Invalid type parameter calling list_dir_or_file!')
            return []
    return list_result


def gen_dict_and_store(name='undefined', images_path = [], class_index = []):
    if not os.path.exists(mat_base_dir):
        os.mkdir(mat_base_dir)
    file_name = os.path.join(mat_base_dir, name + '.mat')
    sio.savemat(file_name, {'images': images_path, 'labels': class_index}, format='5')


def gen_train_valid(prefix='_ps', ratio=0.75):
    process_train(prefix=prefix, valid=True, valid_ratio=ratio)


def process_soe_ratio():
    valid_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    name_str = ['p90', 'p80', 'p70', 'p60', 'p50', 'p40', 'p30', 'p20', 'p10']
    for vi in range(len(valid_ratios)):
        for i in range(0, 10):
            train_image_path, train_indexes,_, _ = process_train(class_names=class_name,base_path=train_base_dir,
                                                                 prefix="", save_mat=False, valid=True, valid_ratio=valid_ratios[vi])
            name = 'train_' + name_str[vi]+ '_' +str(i)
            gen_dict_and_store(name=name, images_path=train_image_path, class_index=train_indexes)


def process_soe_all_set():
    process_train()
    process_test_set()


def process_eoc2_set():
    class_names = {'BMP2': 0, 'BRDM_2': 1, 'BTR70': 2, 'T72': 3}
    process_train(class_names=class_names, base_path=train_base_dir, prefix='_eoc2')
    process_eoc2(base_path=eoc2_base_dir, sub_dir_names=['S7', 'A32', 'A62', 'A63', 'A64'], name='_eoc2_cv', labels=[3, 3, 3, 3, 3])
    process_eoc2(base_path=eoc2_base_dir, sub_dir_names=['9566', 'C21', '812', 'A04', 'A05', 'A07', 'A10'], name='_eoc2_vv', labels=[0, 0, 3, 3, 3, 3, 3])


def process_eoc2(base_path=eoc2_base_dir,sub_dir_names=['S7','A32','A62','A63','A64'], name='_eoc2_cv', labels=[3,3,3,3,3], save_mat=True):

    test_image_path = []
    test_indexes = []

    for sub_index in range(len(sub_dir_names)):
        sub_dir = sub_dir_names[sub_index]
        label = labels[sub_index]
        list_file = list_dir_or_file(type='file', base_path=base_path+sub_dir) ; cur_file_num = len(list_file)
        list_file = np.array(list_file)

        test_index = (np.ones(shape=[cur_file_num]) * (label))
        test_indexes.append(test_index)
        test_image_path.append(list_file[:])
        if verbose:
            print(str(label) + ' ' + sub_dir + ' test ' + str(cur_file_num))

    if save_mat:
        gen_dict_and_store(name='test' + name, images_path=test_image_path,
                           class_index=test_indexes)

    return test_image_path, test_indexes


def process_test_set(base_path=test_base_dir, save_name='', class_names=class_name, save_mat=True):
    list_dir = list_dir_or_file(type='dir', base_path=base_path)

    test_image_path = []
    test_indexes = []

    for dir_index in range(len(list_dir)):
        dir_item = list_dir[dir_index]
        name = dir_item[dir_item.rfind('/') + 1:]
        if name not in class_names:
            continue
        label_index = class_names[name]
        list_file = list_dir_or_file(type='file', base_path=dir_item) ; cur_file_num = len(list_file)
        list_file = np.array(list_file)

        test_index = (np.ones(shape=[cur_file_num]) * (label_index))
        test_indexes.append(test_index)
        test_image_path.append(list_file[:])


        if verbose:
            print(str(label_index) + ' ' + dir_item + ' test ' + str(cur_file_num))
    if save_mat:
        gen_dict_and_store(name='test' + save_name, images_path=test_image_path, class_index=test_indexes)


    return test_image_path, test_indexes


def process_train(class_names=class_name, base_path=train_base_dir, prefix="", save_mat=True, valid=False, valid_ratio=0.5, random_flag=False):
    # read all dir
    list_dir = list_dir_or_file(type='dir', base_path=base_path)
    train_image_path = [] ; valid_image_path = []
    train_indexes = [] ;  valid_indexes = []
    for dir_index in range(len(list_dir)):
        dir_item = list_dir[dir_index]
        name = dir_item[dir_item.rfind('/')+1:]
        if name not in class_names:
            continue
        label_index = class_names[name]
        list_file = list_dir_or_file(type='file', base_path=dir_item) ; cur_file_num = len(list_file)
        if random_flag:
            random_index = np.arange(cur_file_num)
            random_obj.shuffle(random_index)
            list_file = list_file[random_index]
        if valid:
            train_num = math.ceil(cur_file_num * (1-valid_ratio))
            train_list = list_file[:train_num] ; train_image_path.append(train_list)
            train_index = (np.ones(shape=[train_num]) * (label_index)).tolist() ; train_indexes.append(train_index)

            valid_num = cur_file_num -train_num
            valid_list = list_file[train_num:] ; valid_image_path.append(valid_list)
            valid_index = (np.ones(shape=[valid_num]) * (label_index)).tolist() ; valid_indexes.append(valid_index)
        else:
            train_index = (np.ones(shape=[cur_file_num]) * (label_index)).tolist()
            train_indexes.append(train_index)
            train_image_path.append(list_file)

        if verbose:
            if valid:
                print(str(label_index) + ' ' + dir_item+': train ' + str(train_num))
                print(str(label_index) + ' ' + dir_item + ': valid ' + str(valid_num))
            else:
                print(str(label_index) + ' ' + dir_item+': train ' + str(cur_file_num))

        if verbose:
            print('--------------------------')

    if save_mat:
        if valid:
            gen_dict_and_store(name='train' + prefix, images_path=train_image_path, class_index=train_indexes)
            gen_dict_and_store(name='valid' + prefix, images_path=valid_image_path, class_index=valid_indexes)
        else:
            gen_dict_and_store(name='train' + prefix, images_path=train_image_path, class_index=train_indexes)

    return train_image_path, train_indexes, valid_image_path, valid_indexes


gen_train_valid()
process_soe_all_set()
process_eoc2_set()
process_soe_ratio()
