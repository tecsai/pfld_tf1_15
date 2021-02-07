# -*- coding: utf-8 -*-
"""
@author: Derek Sai
@time:2021.01.15
"""
import numpy as np
import tensorflow as tf
import cv2
import os
import glob
import h5py
slim = tf.contrib.slim
 
def read_all_path(path):
    """
    获取文件夹下所有图片的路径
    :path 图片所在路径
    :return路径列表
    """
    mark_path = []
    for filename in glob.glob(path):
        # cv2.imread(filename,1) # jpg
        f, ext = os.path.splitext(filename)
        mark = f + '.pts'  # land_mark
        mark_path.append(mark)
    return mark_path
 
def readLmk(fileName):
    """
    获取标注文件的关键点
    :param fileName: 标注文件全路径
    :return: list--关键点列表
    """
    landmarks = []
    if not os.path.exists(fileName):
        return landmarks
    else:
        fp = open(fileName)
        i = 0
        for line in fp.readlines():
            # print line.strip("\n")
            TT = line.strip("\n")
            if i > 2 and i <= 70:
                # print(TT)
                TT_temp = TT.split(" ")
                # print(TT_temp)
                x = float(TT_temp[0])
                y = float(TT_temp[1])
                landmarks.append((x, y))
            i += 1
    return landmarks
 
def get_data_and_label(path):
    """
    获取图片路径下的Image和关键点
    :param path: path
    :return: Image,land_mark
    """
    for filename in glob.glob(path):
        Image = cv2.imread(filename)
        f, ext = os.path.splitext(filename)
        mark_path = f + '.pts'  # land_mark
        landmarks = readLmk(mark_path)
        #####
        # for point in landmarks:
        #     # print('point:', point)
        #     cv2.circle(Image, point, 1, color=(255, 255, 0))
        # cv2.imshow('image', Image)
        #####
        # print(type(landmarks))
        landmarks = np.array(landmarks)
        # print('landmark:', landmarks)
        # print('landmarks_shape:',landmarks.shape)
        landmark = np.reshape(landmarks, [1, 136])
        # print('finally_landmark_shape:',landmark.shape)
        # cv2.waitKey(200)
    # print('---***---' * 5)
    return Image, landmark
 
def showlandmark(image_path, image_label):
    """
    根据图片路径和标注信息路径--显示标注和图像是否统一
    :param image_path:
    :param image_label:
    """
    img = cv2.imread(image_path, 1)
    # width, height, c = img.shape
    # print('image_shape:', width, height, c)
    labelmarks = readLmk(image_label)
    print('关键点个数：', len(labelmarks))
    # for i in range(len(labelmarks)):
    #     print('labelmarks_%s:',i,labelmarks[i])
    #     point_x,point_y = float(labelmarks[i][0]),float(labelmarks[i][1])
    #     print(point_x,point_y)
    #     cv2.drawKeypoints(img,(point_x,point_y),img,color='g')
    for point in labelmarks:
        print('point:', point)
        cv2.circle(img, point, 1, color=(16, 255, 10))
    cv2.imwrite('000.jpg', img)
    cv2.imshow('image_109',img)
    cv2.waitKey(0)
 
 
# def weight_variable(shape, name):
#     """
#     W:权重参数初始化
#     :param shape: w_shape[w,h,c1,C2]
#     :return: W
#     """
#     initial = tf.truncated_normal(shape, stddev=0.01,name=name)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape,name):
#     """
#     b:偏执参数初始化
#     :param shape: w_shape[C2]
#     :return: b
#     """
#     initial = tf.constant(0.01, shape=shape,name=name)
#     return tf.Variable(initial)

def weight_variable(shape, name):
    """
    W:权重参数初始化
    :param shape: w_shape[w,h,c1,C2]
    :return: W
    """
    initial = tf.truncated_normal(shape, stddev=0.01, name=name)
    regularizer = tf.keras.regularizers.l2()
    with tf.variable_scope(name):
        return tf.get_variable("wight", regularizer=regularizer, initializer=initial)


def bias_variable(shape, name):
    """
    b:偏执参数初始化
    :param shape: w_shape[C2]
    :return: b
    """
    initial = tf.constant(0.01, shape=shape, name=name)
    regularizer = tf.keras.regularizers.l2()
    with tf.variable_scope(name):
        return tf.get_variable("bias", regularizer=regularizer, initializer=initial)
 
def conv2d(x, W,padding,strides=[1,2,2,1]):
    """定义卷积运算
    :param x: input
    :param W: W
    :param padding:填充方式
    :param strides: strides
    :return:
    """
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)


# =========================================================================== #
# Convolutional layer.
# =========================================================================== #
def Conv2d(name, inputs, shape, padding, strides=[2, 2],  bn=False, is_training=True, data_format='NHWC', activation=tf.nn.relu):
    """
        name: variable scope name
        inputs: input tensor
        shape: [kernel_h_size, kernel_w_size, in_channels, out_channels]
        padding: 不区分大小写
        strides: 包含一个或两个整数，表示横向和纵向的步长
        bn: use BN or not
        is_training: training or inference
        data_format: "NHWC" or "NCHW"
        activation: activation function
    """
    kernel_h_size = shape[0]
    kernel_w_size = shape[1]
    in_channels = shape[2]
    out_channels = shape[3]
    if data_format == 'NHWC':
        conv_data_format = 'channels_last'
    else:
        conv_data_format = 'channels_first'
    with tf.variable_scope(name) as scope:
        conv_out = tf.layers.conv2d(inputs=inputs, filters=out_channels, kernel_size=[kernel_h_size, kernel_w_size],
                                    strides=strides, use_bias=not bn,
                                    trainable=is_training, data_format=conv_data_format,
                                    padding=padding, activation=activation)

    if bn == True:
        conv_out = tf.layers.batch_normalization(conv_out, is_training=is_training)

    return conv_out


# =========================================================================== #
# Deepwise convolution layer.
# =========================================================================== #
def DW_Conv2d(x,W):
    """反卷积运算
    :param x: input
    :param W: W
    :return:
    """
    return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def MaxPool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):
    """最大池化操作
    :param x: x
    :return:
    """
    return tf.nn.max_pool(x, ksize=ksize,
                          strides=strides, padding=padding)
 
def Batch_Norm(value,is_training=True):
    '''
        批量归一化  返回批量归一化的结果
        value:代表输入，第一个维度为batch_size
        is_training:当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，
        这样就会使用训练样本集的均值和方差。默认测试模式
        name：名称。
    '''
    if is_training is True:
        # 训练模式 使用指数加权函数不断更新均值和方差
        return tf.layers.batch_normalization(value, training=True)
    else:
        # 测试模式 不更新均值和方差，直接使用
        return tf.layers.batch_normalization(value, training=False)
 
 
def make_bottleneck_block(inputs, expantion, depth, stride,kernel=(3, 3)):
    """从块定义构造瓶颈块--Construct a bottleneck block from a block definition.
    There are three parts in a bottleneck block:bottleneck block包含三个部分
    1. 1x1 pointwise convolution with ReLU6, expanding channels to 'input_channel * expantion'
    2. 3x3 depthwise convolution with ReLU6
    3. 1x1 pointwise convolution
    """
    # The depth of the block depends on the input depth and the expantion rate.
    input_depth = inputs.get_shape().as_list()[-1] # input_chanel
    # filter1 = ()
    block_depth = input_depth * expantion # 扩展通道
    # First do 1x1 pointwise convolution,relu6
    inputs = Batch_Norm(inputs, is_training=True)
 
    block1 = tf.layers.conv2d(inputs=inputs, filters=block_depth, kernel_size=[1, 1],
                              padding='same', activation=tf.nn.relu6)
    # Second, do 3x3 depthwise convolution,relu6
    # filter2 = (3,3,deep,1)
    depthwise_kernel = tf.Variable(
        tf.truncated_normal(shape=[kernel[0], kernel[1], block_depth, 1], stddev=0.001))

    block2 = tf.nn.depthwise_conv2d_native(input=block1, filter=depthwise_kernel,
                                           strides=stride, padding='SAME')
    block2 = tf.nn.relu6(features=block2)
    # Third, pointwise convolution.
    block3 = tf.layers.conv2d(inputs=block2, filters=depth, kernel_size=[1, 1], padding='SAME')
    if stride[1] == 1:
        last_n_filter = input_depth  # 输入通道数
        if depth > last_n_filter:
            shortcut = tf.layers.conv2d(inputs,depth,1,1)
        elif depth < last_n_filter:
            shortcut = tf.layers.conv2d(inputs, depth, 1, 1)
        else:
            shortcut = inputs
        block3 = tf.add_n([block3,shortcut])
 
    return block3
 
def subsample(inputs, factor, scope=None):
    '''降采样方法：
    factor:采样因子 1：不做修改直接返回 不为1：使用slim.max_pool2d降采样'''
    if factor ==1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)
 
def bottleneck(inputs, expantion, depth, stride,kernel=(3, 3)):
    """
    :param inputs: 输入
    :param expantion: 扩展倍数
    :param depth:最后的通道数
    :param stride: stride
    :param kernel: kernel
    :return:
    """
    input_depth = inputs.get_shape().as_list()[-1]  # 输入通道
    block_depth = input_depth * expantion           # 扩展通道
    # 对输入进行bn
    inputs = slim.batch_norm(inputs, activation_fn=tf.nn.relu,
                             scope='inputs')
    if depth == input_depth:
        '''如果残差单元输入通道数和输出通道数一样使用subsample按步长对inputs进行空间上的降采样'''
        shortcut = subsample(inputs, stride[1], 'shortcut')
    else:
        '''如果残差单元输入通道数和输出通道数不一样，
                    使用stride步长的1x1卷积改变其通道数，使得输入通道数和输出通道数一致'''
        shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride[1],
                               normalizer_fn=None, activation_fn=None,
                               scope='shortcut')
 
    '''定义残差：
    第一步：1x1尺寸、步长为1、输出通道数为depth_bottleneck的卷积
    第二步：3x3尺寸、步长为stride、输出通道数为depth_bottleneck的卷积
    第三步：1x1尺寸、步长为1、输出通道数为depth的卷积'''
    block1 = tf.layers.conv2d(inputs=inputs, filters=input_depth, kernel_size=[1, 1],
                              padding='same', activation=tf.nn.relu6)
    depthwise_kernel = tf.Variable(
        tf.truncated_normal(shape=[kernel[0], kernel[1], input_depth, expantion], stddev=0.01))
    block2 = tf.nn.depthwise_conv2d_native(input=block1, filter=depthwise_kernel,
                                           strides=stride, padding='SAME')
    block2 = tf.nn.relu6(features=block2)
    block3 = tf.layers.conv2d(inputs=block2, filters=input_depth, kernel_size=[1, 1], padding='SAME')
    output = shortcut + block3
 
    return output
 
def get_train_and_label(train_path):
    with h5py.File(train_path, 'r') as f:
        images = f['Images'][:]
        labels = f['landmarks'][:]
        oulaTheta = f['oulaTheta'][:]
    X_train = images
    Y_train = labels
    Y_theta = oulaTheta
    # print(X_train.shape) # (3111,112,3)
    #     # print(Y_train.shape) # (3111,136)
    #     # print(Y_theta.shape)
    return X_train,Y_train,Y_theta
 
def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    # offsets:x1-x2//2 ==>中间部分
    #begin = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, x1_shape[3]]
    begin = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    # size = [-1, x2_shape[1], x2_shape[2], x1_shape[3]]
    # 从x1中抽取部分内容，begin:n维列表，begin[i]表示从inputs中第i维抽取数据时，
    # 相对0的起始偏移量，也就是从第i维的begin[i]开始抽取数据
    # size：n维列表，size[i]表示要抽取的第i维元素的数目
    x1_crop = tf.slice(x1, begin, size)
    return tf.concat([x1_crop, x2], axis=3)
 
 
# # readLmk(image_path)
# # showlandmark(image_path,image_label)