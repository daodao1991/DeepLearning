#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import tensorflow as tf

#parameters
IMAGE_SIZE = 28   #图片维度
NUM_CHANNELS = 1  #单通道
CONV1_SIZE = 5    #第一个卷积层的维度
CONV1_KERNEL_NUM = 32  #第一层中卷积核的个数
CONV2_SIZE = 5    #第二个卷积层的维度
CONV2_KERNEL_NUM = 64  #第二层中卷积核的个数
FC_SIZE = 512     #全连接层的节点个数
OUTPUT_NODE = 10  #输出层中节点个数，即最终分为多少类

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)
