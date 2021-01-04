import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from vgg16 import vgg_model as model

means = [123.68, 116.779, 103.939]  # VGG训练时图像预处理减去的均值
x = tf.placeholder(tf.float32, [None, 224, 224, 3])

sess = tf.Session()
vgg = model.vgg16(x)
fc8_finetuining = vgg.probs
# 拿到训练好的模型
saver = tf.train.Saver()
print("model restoring")
saver.restore(sess, "./vgg_model/")

real_answer = ["dog", "dog", "dog", "dog", "cat",
               "cat", "cat", "cat", "cat", "cat",
               "cat", "dog", "cat", "cat", "cat",
               "cat", "dog", "dog", "cat", "cat"]
for i in range(1, 21):

    filepath = './img_data/vgg16/data/test1/' + str(i) + '.jpg'
    img = imread(filepath, mode="RGB")
    img = imresize(img, (224, 224))
    img = img.astype(np.float32)

    for c in range(3):
        img[:, :c] -= means[c]
    prob = sess.run(fc8_finetuining, feed_dict={x: [img]})
    max_index = np.argmax(prob)
    if max_index == 0:
        max_index = "cat"
    elif max_index == 1:
        max_index = 'dog'

    print("this picture ", i, "prediction is ", max_index, "correct is :", real_answer[i - 1])