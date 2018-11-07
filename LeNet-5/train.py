#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

#加载自定义的前向传播（函数和常量）
import inference
import numpy as np

#配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULRAZTION_RATE = 0.0001 #Regularization_Rate表示正则项在loss function中所占的比重
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

#模型保存的路径和文件名
MODEL_SAVE_PATH = os.getcwd() + "/model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    # 定义输入为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, 
            [BATCH_SIZE,
            inference.IMAGE_SIZE,
            inference.IMAGE_SIZE,
            inference.NUM_CHANNELS],
            name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULRAZTION_RATE)
    #使用定义好的前向传播
    y = inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)
   
    #定义损失函数、学习率、滑动平均操作并训练
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = ema.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, 
            global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    #初始化tensorflow持久化类
    saver = tf.train.Saver() 
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #在训练过程中不测试模型在验证集中的表现，验证和测试的过程由独立程序执行
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,
                    [BATCH_SIZE,
                    inference.IMAGE_SIZE,
                    inference.IMAGE_SIZE,
                    inference.NUM_CHANNELS])
            #zzz是一个占位符，无论是什么都可以。之前用 _ 下划线代替的
            zzz,loss_value, step = sess.run([train_op, loss, global_step],
                    feed_dict ={x:reshaped_xs, y_:ys})

            #每1000轮保存一次模型
            if i % 1000 == 0:
                print("After %d train step, loss on trainable  batch is  %g." % (step, loss_value))
                saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME, global_step = global_step)


def main(argv=None):
    mnist = input_data.read_data_sets(os.getcwd() + "MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
