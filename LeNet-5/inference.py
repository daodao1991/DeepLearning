#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf
import os

#定义神经网络的参数 784个输入特征 10中分类 中间层有500个结点
INPUT_NODE = 784
OUT_NODE = 10

#传入的图片数据 28*28*1 的三维矩阵  标签为10维矩阵
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
#第一层卷积神经网络的深度和尺寸
CONV1_DEEP = 32
CONV1_SIZE = 5
#第二层卷积神经网络的深度和尺寸
CONV2_DEEP = 64
CONV2_SIZE = 5
#全连接层节点个数
FC_SIZE = 512

#LAYER1_NODE = 500

# 定义卷积神经网络的前向传播过程。这里添加了一个新的参数train，用于区别训练过程和测试过程。在这个程序  中将用到dropout方法
# dropout可以进一步提升模型可靠性并防止过拟合（dropout过程只在训练时使用）
def inference(input_tensor, train, regularizer):

    # 声明第一层卷积层的变量并实现前向传播过程。
    # 通过使用不同层的命名空间来隔离不同层的变量，让每一层中的变量命名只需要考虑在当前层的作用，
    # 不需要担心重名的问题。
    # 和标准LeNet-5模型不大一样，这里定义卷积层输入28 * 28 * 1 的原始MNIST图片像素。
    # 卷积层中使用全0填充，输出28 * 28 * 32 矩阵
    with tf.variable_scope('layer1_conv1'): #通过tf.get_variable()为变量名指定命名空间
        conv1_weights = tf.get_variable(
                'weight',[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS,CONV1_DEEP],
                initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
                'bias',[CONV1_DEEP],initializer = tf.constant_initializer(0.0))

        # 使用边长5， 深度32的过滤器，移动步长1，使用全0填充
        conv1 = tf.nn.conv2d(
                input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播过程。使用最大池化，过滤器边长2，使用全0填充，移动步长2.
    # 输入 28 * 28 * 32   输出  14 * 14 * 32
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(
                relu1, ksize=[1,2,2,], strides=[1,2,2,1], padding='SAME')

    # 声明第三层卷积层的变量实现前向传播过程。
    # 输入 14 * 14 * 32  输出 14 * 14 * 64
    with tf.variable_scope('layer3_conv2'):
        conv2_weights = tf.get_variable(
                'weight', [CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(
                'bias',[CONV2_DEEP], initializer = tf.constant_initializer(0.0))

        # 使用边长5， 深度64的过滤器，移动步长1， 全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        
    # 实现第四层池化层的前向传播过程。
    # 输入14 * 14 * 64  输出 7 * 7 * 64
    with tf.name_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(
                relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # 将第四层池化层的输出转化为第五层全连接层的输入格式。
    # 将 7 * 7 * 64 的矩阵拉直成一个向量。
    # pool2.get_shape()函数返回的是一个元组，可以等到输出矩阵维度，通过as_list()操作转换成list
    # 每一层神经网络的输入输出都是一个batch的矩阵，这里得到的维度包含一个batch数据的个数
    pool_shape = pool2.get_shape().as_list()

    # 计算将矩阵拉直成向量之后的长度，就是矩阵长宽与深度的乘积。
    # pool_shape[0]是一个batch中数据的个数。
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # 通过tf.reshape函数将第四层的输出变成一个batch的向量。
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])   #[1, 7X7X64]

    # 声明第五层全连接层的变量并实现前向传播过程.
    # 3136->512  输入是拉直后的向量，长度3136.  输出向量，长度512
    # 使用了dropout. dropout一般在全连接层使用.
    with tf.variable_scope('layer5_fc1'):
        fc1_weights = tf.get_variable(
                'weight', [nodes, FC_SIZE], 
                initializer = tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        # 正则化的目的是为了防止过拟合
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
            #将权重矩阵fc1_weights所对应的正则项加入到集合losses中 
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

	#train 为 传入决定是否用dropout的参数
        if train:
            fc1 = tf.nn.dropput(fc1, 0.5)

    # 声明第六层全连接层的变量，并实现前向传播过程
    # 512->10  输出通过softmax 后就得到最后的分类结果
    with tf.variable_scope('Layer6_fc2'):
        fc2_weights = tf.get_variable(
                'weight', [FC_SIZE, NUM_LABELS], initializer =
                tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [NUM_LABELS], initialzier =
                tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
