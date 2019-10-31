# tool script
"""
@author: mengxue.Zhang
"""

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import math
import scipy.io as sio

rd_obj = np.random.RandomState(2019)

data_generator = ImageDataGenerator(
   vertical_flip=False,
   horizontal_flip=False)


def get_batch(type='train', mat_paths='./input/', image_batch_size=100, shapes=[88, 88]):
   if type=='train':
      return get_train_batch(mat_paths, image_batch_size, shapes)
   elif type =='valid' or type =='test':
      return get_evaluation_batch(mat_paths, image_batch_size, shapes)
   elif type=='test_gt':
      return get_test_gt(mat_paths)
   elif type=='get_images':
       # return each class (image, label)
       return get_images(mat_paths, shapes)
   else:
      print('Invalid type parameter calling get_batch!')


def get_step(mat_path, type='train', image_batch_size=100):
    dict_obj = sio.loadmat(mat_path)
    new_labels = []

    labels = dict_obj['labels'][0]
    [new_labels.extend(label[0].tolist()) for label in labels]
    new_labels = np.array(new_labels)
    if type == 'train':
        step = math.ceil(new_labels.shape[0] / image_batch_size)
    elif type == 'valid' or type == 'test':
        step = math.ceil(new_labels.shape[0] / image_batch_size)
    else:
        print('Invalid type parameter calling get_step!')
        step = 0
    return step


def get_test_gt(mat_paths = ''):
   image_paths, labels = read_mat2_list(mat_paths)
   new_labels = []
   [new_labels.extend(label[0].tolist()) for label in labels]
   new_labels = np.array(new_labels)
   return new_labels

# directly get images
def get_images(mat_paths = '', shapes=[88, 88]):
    image_paths, labels = read_mat2_list(mat_paths)
    new_image_paths = []
    new_labels = []
    [new_image_paths.extend(class_path.tolist()) for class_path in image_paths]
    [new_labels.extend(label[0].tolist()) for label in labels]

    new_image_paths = np.array(new_image_paths);
    new_labels = np.array(new_labels)

    cur_batch_images_path = new_image_paths; cur_labels = new_labels
    batch_images = batch_imread_image(cur_batch_images_path, shapes=shapes)
    cur_labels = np.expand_dims(cur_labels, -1)
    batch_images = pre_process_data(batch_images)
    return (batch_images, cur_labels)


def pre_process_data(images):
    return images / 255.0


def read_mat2_list(mat_path):
    dict_obj = sio.loadmat(mat_path)
    return dict_obj['images'][0], dict_obj['labels'][0]


def batch_imread_image(batch_image_paths, channel=1, resize=True, shapes=[88, 88]):
   # single thread read
    flag = True
    for image_path in batch_image_paths:
        image_path = image_path.rstrip()
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (shapes[0], shapes[1]), interpolation=cv2.INTER_LINEAR)
            if channel == 1:
                image = image[:, :, 0]
                image = np.expand_dims(image, axis=2)
            else:
                image = image[:, :, (2, 1, 0)]
        except:
            print(image_path+' image file is broken!')

        image = np.expand_dims(image, axis=0)
        images = image if flag else np.concatenate((images, image), axis=0)
        flag = False

    images = np.array(images)
    return images


def get_train_batch(mat_paths='', labeled_batch_size=100, shapes=[88, 88]):
   image_paths, labels = read_mat2_list(mat_paths)
   new_image_paths = []; new_labels = []
   [new_image_paths.extend(class_path.tolist()) for class_path in image_paths]
   [new_labels.extend(label[0].tolist()) for label in labels]
   new_image_paths = np.array(new_image_paths) ; new_labels = np.array(new_labels)
   # scope infinitly
   while True:
      random_indexes = np.arange(len(new_image_paths))
      rd_obj.shuffle(random_indexes)
      new_image_paths = new_image_paths[random_indexes]
      new_labels = new_labels[random_indexes]
      for batch_end_index in range(0, new_image_paths.shape[0], labeled_batch_size):
         cur_batch_images_path = new_image_paths[batch_end_index:min(batch_end_index+labeled_batch_size,new_image_paths.shape[0])]
         batch_images = batch_imread_image(cur_batch_images_path, resize=False, shapes=shapes)
         cur_labels = new_labels[batch_end_index:min(batch_end_index+labeled_batch_size, new_labels.shape[0])]
         batch_images = pre_process_data(batch_images)
         batch_images, cur_labels = get_more_images(batch_images, cur_labels, final_image_num=labeled_batch_size,
                                                    use_patch=True, each_img_need_num=1)
         cur_labels = np.expand_dims(cur_labels, -1)

         yield (batch_images, cur_labels)


def get_evaluation_batch(mat_paths = '', labeled_batch_size=100, shapes=[88, 88], random_flag=False):
   image_paths, labels = read_mat2_list(mat_paths)
   new_image_paths = []; new_labels = []
   [new_image_paths.extend(class_path.tolist()) for class_path in image_paths]
   [new_labels.extend(label[0].tolist()) for label in labels]
   new_image_paths = np.array(new_image_paths) ; new_labels = np.array(new_labels)
   # scope infinitly
   while True:
      if random_flag:
         random_indexes = np.arange(len(new_image_paths));
         rd_obj.shuffle(random_indexes)
         new_image_paths = new_image_paths[random_indexes];
         new_labels = new_labels[random_indexes]

      for batch_end_index in range(0, new_image_paths.shape[0], labeled_batch_size):
         cur_batch_images_path = new_image_paths[batch_end_index:min(batch_end_index+labeled_batch_size, new_image_paths.shape[0])]
         batch_images = batch_imread_image(cur_batch_images_path, shapes=shapes)
         cur_labels = new_labels[batch_end_index:min(batch_end_index+labeled_batch_size, new_labels.shape[0])]
         cur_labels = np.expand_dims(cur_labels, -1)
         batch_images = pre_process_data(batch_images)
         yield (batch_images, cur_labels)


def get_more_images(imgs, labels, final_image_num=100, use_patch=False, each_img_need_num=1, image_shape=[88, 88]):

   def gen_more_images(images, labels, batch_size=100):
      return next(data_generator.flow(images, labels, batch_size=batch_size))

   if use_patch:
       imgs, patch_nums = gen_patch_images(imgs, blank_cut=[0, 0, 0, 0], patch_hw=[image_shape[0], image_shape[0]],
                                           resize=False, each_img_need_num=each_img_need_num,
                                           random=rd_obj)
       labels = np.repeat(labels, each_img_need_num)

   batch_images, cur_labels = gen_more_images(imgs, labels, final_image_num)
   return batch_images, cur_labels


def gen_patch_images(origin_imgs, blank_cut=[0, 0, 0, 0], patch_hw=[88, 88], resize=True, each_img_need_num=10000, random=rd_obj):
    shape_length = len(origin_imgs.shape)
    assert(shape_length == 4)
    flag = True
    for origin_img in origin_imgs:
        gen_patch_imgs, gen_num = gen_patch_image(origin_img, blank_cut=blank_cut, patch_hw=patch_hw,
                                                  resize=resize, need_num=each_img_need_num, random=rd_obj)

        final_patchs = gen_patch_imgs if flag else np.concatenate([final_patchs, gen_patch_imgs], axis=0)
        flag = False

        gen_nums = gen_num * origin_imgs.shape[0]

    return final_patchs, gen_nums


# blank_cut [top, bottom, left, right] contain=False
def gen_patch_image(origin_img, blank_cut=[0, 0, 0, 0], patch_hw=[88, 88], resize=True, need_num=10000,
                    random=rd_obj):
    def gen_all_border(patch_hw, max_hw, cut_hw = [5,5]):
        delta_h = max_hw[0] - patch_hw[0]
        delta_w = max_hw[1] - patch_hw[1]
        borders = []
        for t in range(0, delta_h+1):
            for l in range(0, delta_w+1):
                border = [t + cut_hw[0], l + cut_hw[1], t + patch_hw[0] + cut_hw[0], l + patch_hw[1] + cut_hw[1]]
                borders.append(border)
        return borders

    shape_length = len(origin_img.shape)
    if shape_length == 2 or shape_length == 3:
        height = origin_img.shape[0]
        width = origin_img.shape[1]
        max_hw = [0, 0]
        max_hw[0] = height - blank_cut[0] - blank_cut[1]
        max_hw[1] = width - blank_cut[2] - blank_cut[3]
        assert (max_hw[0] >= patch_hw[0]) ; assert (max_hw[1] >= patch_hw[1])
        borders = gen_all_border(patch_hw, max_hw, [blank_cut[0], blank_cut[2]])
        gen_num = len(borders)
        if need_num >= gen_num:
            need_num = gen_num

        random_indexes = np.arange(gen_num) ; random.shuffle(random_indexes)
        random_indexes = random_indexes[0:need_num]
        borders = np.array(borders)
        borders = borders[random_indexes]

        flag = True
        for border_index in range(len(borders)):
            border = borders[border_index]
            patch_img = origin_img[border[0]:border[2], border[1]:border[3]]
            if resize:
                patch_img = cv2.resize(patch_img, (height, width), interpolation=cv2.INTER_LINEAR)

            patch_img = np.expand_dims(patch_img, axis=0)

            gen_patch_imgs = patch_img if flag else np.concatenate([gen_patch_imgs, patch_img], axis=0)
            flag = False

        return gen_patch_imgs, need_num
    else:
        # more one images
        print('Origin img must be one image')
        return origin_img, 1


size = [88, 88, 1]
def add_noise(batch_images, ratio=0.01):
    noised_num = math.ceil(ratio * size[0] * size[1])
    idx = np.arange(size[0]*size[1])

    for i in range(batch_images.shape[0]):
        flags = np.zeros(shape=size[0] * size[1])
        uniform_noise = rd_obj.uniform(low=0.0, high=1.0, size=size)
        rd_obj.shuffle(idx)
        flags[idx[0:noised_num]] = 1
        flags = np.reshape(flags, size)
        batch_images[i] = np.multiply(batch_images[i], 1-flags) + np.multiply(uniform_noise, flags)

    return batch_images

def get_noised_batch(test_dir, ratio=0.01):
    batch = get_evaluation_batch(test_dir)
    for batch_images, batch_labels in batch:
        batch_images = add_noise(batch_images, ratio)
        yield batch_images, batch_labels

