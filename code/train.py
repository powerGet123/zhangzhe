import os
import tensorflow as tf
from time import time
import utils
from vgg16 import vgg_model as model

startTime = time()
batch_size = 16
capacity = 256  # 内存中最大数据容量
means = [123.68, 116.779, 103.939]  # VGG训练时减去的均值
img_dir = "./img_data/vgg16/data/train"

xs, ys = utils.get_file(img_dir)
image_batch, label_batch = utils.get_batch(xs, ys, 224, 224, batch_size, capacity)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, 2])

vgg = model.vgg16(x)  # 调用构造函数，传入X变量
fc8_finetuining = vgg.probs  # softmax(fc8)
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc8_finetuining,
                                                                       labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_function)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

vgg.load_weights('./vgg16/vgg16_weights.npz', sess)  # 装载VGG16参数

saver = tf.train.Saver()

# 使用协调器Coordinator来管理线程
coord = tf.train.Coordinator()  # 获取到协调器
threads = tf.train.start_queue_runners(coord=coord, sess=sess)  # 协调器启动队列，将线程激活

epoch_start_time = time()

for i in range(100):
    # 取出的标签值是图像名字上截取下来的
    images, labels = sess.run([image_batch, label_batch])
    # 对标签进行one_hot编码
    labels = utils.onehot(labels)
    _, loss = sess.run([optimizer, loss_function], feed_dict={x: images, y: labels})

    epoch_end_time = time()
    print("Current epoch takes {} s,".format(epoch_end_time - epoch_start_time))
    print("loss is %f." % loss)
    epoch_start_time = epoch_end_time

    if (i + 1) % 500 == 0:
        saver.save(sess, os.path.join("./vgg_model/", "epoch{:06}.ckpt".format(i)))
saver.save(sess, "./vgg_model/")
sess.close()
print("epoch finished!")
coord.request_stop()  # 线程关闭
coord.join(threads)  # 所有线程关闭后协调器注销
