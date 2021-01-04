import tensorflow as tf
import numpy as np
# 对VGG16进行优化，保留前几层神经网络，通过改造最后一层全连接层的输出层


class vgg16:
    def __init__(self, imgs):
        self.parameters = []
        self.imgs = imgs
        self.convlayers()
        # 定义卷积网络和全连接层网络
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc8)

    # 存储器
    def saver(self):
        return tf.train.Saver()

    # 最大池化 4个维度，代表num,width,high,channel的移动数量[1,2,2,1] k-size,stride,尺寸减半
    def maxpool(self, name, input_data):
        out = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        return out

    # 卷积函数
    def conv(self, name, input_data, out_channel, trainable=False):
        in_channel = input_data.get_shape()[-1]  # 截取第三维数据，通道数
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32, trainable=trainable)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)  # 加上偏置项
            out = tf.nn.relu(res, name=name)  # 非线性激活函数
        self.parameters += [kernel, biases]
        return out

    # 全连接函数
    def fc(self, name, input_data, out_channel, trainable=False):
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_data, [-1, size])  # 转置成一维向量
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights", shape=[size, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable(name="biases", shape=[out_channel], dtype=tf.float32, trainable=trainable)
            res = tf.matmul(input_data_flat, weights)  # 这里打错后数据未转化成 4096*1 导致matmul计算失败，未生成全连接层，导致pool5未生成
            out = tf.nn.relu(tf.nn.bias_add(res, biases))
        self.parameters += [weights, biases]
        return out

    # 卷积层
    def convlayers(self):
        self.conv1_1 = self.conv("conv1_1", self.imgs, 64, trainable=False)  # trainable参数变动
        self.conv1_2 = self.conv("conv1_2", self.conv1_1, 64, trainable=False)  # trainable参数变动
        self.pool1 = self.maxpool("poolre1", self.conv1_2)

        self.conv2_1 = self.conv("conv2_1", self.pool1, 128, trainable=False)  # trainable参数变动
        self.conv2_2 = self.conv("convwe2_2", self.conv2_1, 128, trainable=False)  # trainable参数变动
        self.pool2 = self.maxpool("pool2", self.conv2_2)

        self.conv3_1 = self.conv("conv3_1", self.pool2, 256, trainable=False)  # trainable参数变动
        self.conv3_2 = self.conv("convrwe3_2", self.conv3_1, 256, trainable=False)  # trainable参数变动
        self.conv3_3 = self.conv("convrew3_3", self.conv3_2, 256, trainable=False)  # trainable参数变动
        self.pool3 = self.maxpool("poolre3", self.conv3_3)

        self.conv4_1 = self.conv("conv4_1", self.pool3, 512, trainable=False)  # trainable参数变动
        self.conv4_2 = self.conv("convrwe4_2", self.conv4_1, 512, trainable=False)  # trainable参数变动
        self.conv4_3 = self.conv("convrwe4_3", self.conv4_2, 512, trainable=False)  # trainable参数变动
        self.pool4 = self.maxpool("pool4", self.conv4_3)

        self.conv5_1 = self.conv("conv5_1", self.pool4, 512, trainable=False)  # trainable参数变动
        self.conv5_2 = self.conv("convrew5_2", self.conv5_1, 512, trainable=False)  # trainable参数变动
        self.conv5_3 = self.conv("conv5_3", self.conv5_2, 512, trainable=False)  # trainable参数变动
        self.pool5 = self.maxpool("poolwel5", self.conv5_3)

    # 全连接层
    def fc_layers(self):
        self.fc6 = self.fc("fc1", self.pool5, 4096, trainable=False)  # trainable参数变动
        self.fc7 = self.fc("fc2", self.fc6, 4096, trainable=False)  # trainable参数变动
        self.fc8 = self.fc("fc3", self.fc7, 2, trainable=True)  # 这是一个二分类问题所以设置参数为2

    # 载入权重
    def load_weights(self, weight_file, sess):  # 获取权重载入VGG模型
        weights = np.load(weight_file)  # './vgg16/vgg16_weights.npz'
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i not in [30, 31]:  # 剔除不需要载入的层,i,k是枚举变量的索引和值，K是变量的那么name
                print(i,k)
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----weights loaded")

