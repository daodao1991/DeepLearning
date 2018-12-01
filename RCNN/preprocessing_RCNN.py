#!/usr/bin/env python3
#-*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import
import numpy as np
import selectivesearch
import cv2
import config
import os
import random
import math
import sys
import skimage   #scikit-image模块
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    '''
    # 将Image文件给修改成227*227的图片大小（当然，RGB三个频道我们保持不变）
        
    :param in_image: 输入的图片 The image to resize.
    :param new_width: `int`. The image new width.
    :param new_height: `int`. The image new height.
    :param out_image: `str`. If specified, save the image to the given path.
    :param resize_mode: `PIL.Image.mode`. The resizing mode.
    :return: The resize image.
    '''
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img


def clip_pic(img, rect):
    '''修剪图片

    Parameters
    ------------
        img: 输入的图片
        rect: rect矩形框的4个参数
    Returns
    ------------
        输入的图片中相对应rect位置的部分与矩形框的一对对角点和长宽信息
    '''
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]


def view_bar(message, num, total):
    '''进度条工具

    Parameters
    ------------
        message: 在进度条前所要显示的信息
        num:     当前已经处理了的对象的个数
        total:   要处理的对象的总的个数
    '''
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)  # ceil(x)返回一个大于或等于x的最小整数
    r = '\r%s:[%s%s]%d%%\t%d%d' % (message, ">"*rate_num, " "*(40-rate_num), rate_nums, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()


def show_rectangle(img_path, regions):
    '''显示矩形框
    Parameters
    ------------
        img_path: 要显示的原图片
        regions:  要在原图片上标注的矩形框的参数
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)  # imshow()用于绘制热图
    for x, y, w, h in regions:
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


# IOU Part 1
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    '''判断两个方框是否有交集，如果有交集返回相交部分的面积'''
    if_intersect = False
    # 通过四条if来查看两个方框是否有交集。如果四种状况都不存在，我们视为无交集
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return False
    # 在有交集的情况下，我们通过大小关系整理两个方框各自的四个顶点， 通过它们得到交集面积
    if if_intersect == True:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


# IOU Part 2
def IOU(ver1, vertice2):
    # vertice in four points
    # 整理输入顶点
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3],
            vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    # 如果存在交集，计算IOU
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False


def load_train_proposals(datafile, num_class, save_path, threshold=0.5, is_svm=False, save=False):
    ''' Read in data and save data for AlexNet '''
    fp = open(datafile, 'r')
    train_list = fp.readlines()
    for num, line in enumerate(train_list):
        labels = []
        images = []
        tmp = line.strip().split()
        # tmp[0] = image path
        # tmp[1] = label
        # tmp[2] = rectangle vertices
        img = cv2.imread(tmp[0])
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
        # selective_search()的返回值类型如下
        '''
        Returns
        ---------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)])]
        regions : array of dict
            [
                {
                    'rect': (left, top, right, bottom),
                    'size': [...]
                    'labels': [...]
                },
                ...
            ]
        '''
        candidates = set()
        for r in regions:
            # 剔除重复的边框
            if r['rect'] in candidates:
                continue
            # 剔除太小的边框
            if r['size'] < 220:
                continue
            if (r['rect'][2] * r['rect'][3]) < 500:
                continue

            ### 调整大小为227 * 227以进行输入
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
            # 删除空的边框
            if len(proposal_img) == 0:
                continue
            # Ignore things contain 0 or not C contiguous array
            x, y, w, h = r['rect']
            # 长或宽为0的方框，剔除
            if w == 0 or h == 0:
                continue
            # image array的dim里有0的，剔除
            [a, b, c] = np.shape(proposal_img)
            if a == 0 or b == 0 or c == 0:
                continue
            resized_proposal_img = resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
            candidates.add(r['rect'])
            img_float = np.asarray(resized_proposal_img, dtype="float32")
            images.append(img_float)
            
            # IOU
            ref_rect = tmp[2].split(',')
            ref_rect_int = [int(i) for i in ref_rect]
            iou_val = IOU(ref_rect_int, proposal_vertice)
            # labels, let 0 represent default class, which is background
            # 即让0代表背景
            index = int(tmp[1])
            if is_svm:
                if iou_val < threshold:
                    labels.append(0)
                else:
                    labels.append(index)
            else:
                label = np.zeros(num_class + 1)
                if iou_val < threshold:
                    label[0] = 1
                else:
                    label[index] = 1
                labels.append(label)
        view_bar("processing image of %s" % datafile.split('/')[-1].strip(), num+1, len(train_list))
        if save:
            # np.save()将数据以Numpy专用的二进制格式保存在.npy文件中
            # 这种方式建议在不需要看保存文件内容的情况下使用
            np.save((os.path.join(save_path, tmp[0].split('/')[-1].split('.')[0].strip()) + 
                '_data.npy'), [images, labels])
    print(' ')
    fp.close()
    

def load_from_npy(data_set):
    '''从.npy文件中加载数据'''
    images, labels = [], []
    data_list = os.listdir(data_set)
    for ind, d in enumerate(data_list):
        i, l = np.load(os.path.join(data_set, d))
        images.extend(i)
        labels.extend(l)
        view_bar("load data of %s" % d, ind+1, len(data_list))
    print(' ')
    return images, labels
