# train script
"""
@author: mengxue.Zhang
"""

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, LearningRateScheduler
import os
from tools import *
from keras import backend as K
import keras.backend.tensorflow_backend as KTF

import tensorflow as tf
import sklearn.metrics as metric
import pandas as pd

epochs = 300
image_batch_size = 100
model_file_prefix = 'mdl_simple_k0_wght'
model_file_suffix = '.hdf5'
model_file = model_file_prefix + model_file_suffix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def train_model(model_func=None, mode='DEBUG', base_dir='./soe_cp_convnet_rp', train_prefix='train', test_prefix='test',
                multi_file=False, run_times=10, lr=0.001, train_shapes=[88, 88], save_best=0, epochs=epochs, feature_map=False):
    assert model_func
    gpu_setting(init_ratio=0.4)
    verbose = 1 if mode == 'DEBUG' else 0
    n_results = []

    if feature_map:
        run_times = 1

    for rt in range(run_times):

        train_mat = './input/' + train_prefix + '_' + str(rt) + '.mat' if multi_file else \
            './input/' + train_prefix + '.mat'
        test_mat = './input/' + test_prefix + '.mat'

        rt = 'run' + str(rt)
        task_dir = base_dir + '/' + rt + '/'
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        train_gen_batch = get_batch('train', train_mat, image_batch_size, train_shapes)
        train_step = get_step(train_mat, 'train')

        test_gen_batch = get_batch('test', test_mat, image_batch_size)
        test_step = get_step(test_mat, 'test')
        model = model_func.__next__()

        callbacks, MODEL_FILE = get_model_callbacks(save_file=task_dir, save_best=save_best)

        if verbose:
            model.summary()
        model.fit_generator(generator=train_gen_batch,
                            steps_per_epoch=train_step,
                            epochs=epochs,
                            verbose=verbose,
                            validation_data=test_gen_batch,
                            validation_steps=test_step,
                            callbacks=callbacks,
                            shuffle=False)

        model.load_weights(filepath=MODEL_FILE)
        test_gen_batch = get_batch('test', test_mat, image_batch_size)
        test_gt = get_batch(type='test_gt', mat_paths=test_mat)
        results = model.predict_generator(generator=test_gen_batch, steps=test_step, verbose=verbose)
        K.clear_session()
        tf.reset_default_graph()
        model = None
        assert (results.shape[0] == test_gt.shape[0])
        predict = np.argmax(results, axis=1)
        n_results.append(cal_result(predict, test_gt, task_dir))

    if not feature_map:
        mes_str = '  Avg Accuracy: ' + str(np.average(n_results)) + ' Std Accuracy: ' + str(np.std(n_results))
        print(mes_str)
        with open(base_dir + '/' + 'summary.txt', 'w') as f:
            f.write(mes_str)


def load_model(model, model_path, i, model_file=model_file):
    trained_cnn_dir = model_path + '/' + 'run' + str(i) + '/' + model_file
    model.load_weights(trained_cnn_dir)
    return model


def test_noise_model(model_func, model_path, model_num=10, mode='DEBUG', base_dir='./soe_cp_convnet', test_prefix='test',
               ratios=[0, 0.01, 0.05, 0.10, 0.15], run_times=5):#5
    assert model_func
    gpu_setting(init_ratio=0.4)
    verbose = 1 if mode == 'DEBUG' else 0
    num = len(ratios)
    results = np.zeros(shape=[2, num])
    task_dir = base_dir + '/'
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    for ratio_index in range(num):
        result = []
        for i in range(model_num):

            model = model_func.__next__()
            if verbose:
                model.summary()

            model = load_model(model, model_path, i)
            for rt in range(run_times):
                test_mat = './input/' + test_prefix + '.mat'
                test_batch = get_noised_batch(test_mat, ratio=ratios[ratio_index])
                test_step = get_step(test_mat, 'test')
                test_gt = get_batch(type='test_gt', mat_paths=test_mat)
                pred = model.predict_generator(generator=test_batch, steps=test_step, verbose=0)
                assert (pred.shape[0] == test_gt.shape[0])
                pred = np.argmax(pred, axis=1)
                aa = metric.accuracy_score(y_true=test_gt, y_pred=pred, normalize=True)
                result.append(aa)

            K.clear_session()
            tf.reset_default_graph()
            model = None

        results[0, ratio_index] = np.average(np.array(result))
        results[1, ratio_index] = np.std(np.array(result))

    file_name = task_dir + 'summary.txt'
    with open(file_name, 'w') as f:
        for ratio_index in range(num):
            line_str = str(ratios[ratio_index]) + ',' + str(results[0, ratio_index])+ ',' + str(results[1, ratio_index]) + '\n'
            f.write(line_str)


def gpu_setting(init_ratio=0.4):
    # 40%　GPU
    config = tf.ConfigProto(device_count={'gpu':0})
    config.gpu_options.per_process_gpu_memory_fraction = init_ratio
    session = tf.Session(config=config)
    # SET Session
    KTF.set_session(session)


def cal_result(predict, test_gt, cur_dir):
    test_csv = pd.DataFrame()
    test_csv['gt'] = test_gt.astype(np.int32)
    test_csv['pred'] = predict.astype(np.int32)
    csv_file = cur_dir + 'test.csv'
    test_csv.to_csv(csv_file, index=False, float_format='%.6f')
    aa = metric.accuracy_score(y_true=test_gt, y_pred=predict, normalize=True)
    with open(cur_dir + 'test.txt', 'w') as f:
        f.write(str(aa))
    print('        ' + str(aa))
    return aa


def get_model_callbacks(save_file, save_best=0):
    call_backs = []
    MODEL_FILE = save_file + model_file

    if save_best>0:
        call_backs.append(ModelCheckpoint(MODEL_FILE, save_best_only=True, save_weights_only=True, monitor='val_acc'))
    else:
        call_backs.append(ModelCheckpoint(MODEL_FILE, save_weights_only=True))

    def step_decay(epoch, lr):
        return lr * math.exp(-1 * epoch * 0.0001)#1e-4

    call_backs.append(LearningRateScheduler(step_decay))
    call_backs.append(TensorBoard(log_dir=save_file,
                                     histogram_freq=0,
                                     batch_size=image_batch_size,
                                     write_graph=True,
                                     write_grads=False, write_images=False, embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None))
    return call_backs, MODEL_FILE



