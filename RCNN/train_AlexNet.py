#!/usr/bin/env python3
#-*- coding:utf-8 -*-


from __future__ import division, print_function, absolute_import
import pickle  # 用于序列化存储的模块
import numpy as np
import config
import os.path
import codecs  # 用于编码转换的模块
import cv2
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_ normalization
from tflearn.layers.estimator import regresssion
import preprocessing_RCNN as prep


def load_data(datafile, num_class, save=False, save_path='dataset.pkl'):
    fp = codecs.open(datafile, 'r', 'utf-8') # 返回的是unicode
    train_list = fp.readlines()
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split()
        fpath = tmp[0]
        img = cv2.imread(fpath)
        img = prep.resize_image(img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        np_img = np.asarray(img, dtype="float32")
        images.append(np_img)

        index = int(tmp[1])
        label = np.zeros(nums_class)
        label[index] = 1
        labels.append(label)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    fp.close()
    return images, labels


def load_from_pkl(dataset_file):
    X, Y = pickle.load(open(dataset_file, 'rb'))
    return X, Y


# 创建网络 AlexNet
def create_AlexNet(num_classes):
