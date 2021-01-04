import tensorflow as tf
from vgg_preprocess import preprocess_for_train
import os
import numpy as np
from sklearn.utils import shuffle


def get_file(file_dir):
    images = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
    labels = []
    for label_name in images:
        letter = label_name.split("\\")[-1].split('.')[0]
        if letter == "cat":
            labels.append(0)
        else:
            labels.append(1)
    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

# 数据的输入 vgg16使用的图像是224*224的寸尺 VGG19是用的227*227
# img_width = 224
# img_height = 224


# 通过读取列表来载入批量图片及标签
def get_batch(img_list, label_list, img_width, img_height, batch_size, capacity):
    image = tf.cast(img_list, tf.string)
    label = tf.cast(label_list, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    # 以VGG的训练模式继续训练此图像得到相应的特征子
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = preprocess_for_train(image, 224, 224)  # resize=24

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])  # reshape(t,shape)

    return image_batch, label_batch


def onehot(labels):
    n_sample = len(labels)     # 一次读取32张图片
    n_class = max(labels) + 1  # 2个类别

    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1

    return onehot_labels
