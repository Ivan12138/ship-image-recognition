#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：ship-image-recognition
@File ：ship_image_recognition.py
@Author ：wanghao
@Date ：2022/3/19 16:00
@Description: TODO
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


# load数据集数据
from PIL import Image


def loadcifar(filename):
    with open(filename, "rb") as f:
        batch4 = pickle.load(f, encoding='bytes')
    # [b'batch_label', b'labels', b'data', b'filenames']
    # print(batch4.keys())
    data, labels = batch4[b'data'], batch4[b'labels']
    return data, labels


# load label_names
def load_labelsnames(filepath):
    filename = os.path.join(filepath, 'batches.meta')
    with open(filename, 'rb') as f:
        batches_meta = pickle.load(f, encoding='bytes')
    # [b'num_cases_per_batch', b'label_names', b'num_vis']
    # print(batches_meta.keys())
    labels_names = batches_meta[b'label_names']
    return labels_names


# 处理数据集
def process_cifar10(filepath):
    filename = os.path.join(filepath, "data_batch_4")
    data, labels = loadcifar(filename)
    return data, labels

# 将numpy数据格式转换为image格式，以便后续使用plt画出来
def to_pil(data):
    r = Image.fromarray(data[0])
    g = Image.fromarray(data[1])
    b = Image.fromarray(data[2])
    pil_img = Image.merge('RGB', (r, g, b))
    # print('pil_img', pil_img)
    return pil_img
#

# 随机可视化，主要是使用plt将图画出来
def visualable(imgs, labels, labels_names):
    """

    :param imgs:变形后的data
    :param labels: 所有数据的labels[0,3,4,5,6,3,2,2]
    :param labels_names: 标签names b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']
9
    :return: None, 直接画图
    """
    figure = plt.figure(figsize=(len(labels_names), 10))

    # 所有图像数目
    idxs = list(range(len(imgs)))
    # print(idxs)
    np.random.shuffle(idxs)
    # 命名个数
    count = [0] * len(labels_names)
    # print(count)  #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for idx in idxs:
        label = labels[idx]
        # ???
        # print(label)
        print(count[label])
        if count[label] >= 10:
            continue
        if sum(count) > 10 * len(labels_names):
            break
        img = to_pil(imgs[idx])
        label_name = labels_names[label]

        subplot_idx = count[label] * len(labels_names) + label +1
        # 画出对应标签与子图序号

        # print(f"对应标签：{label} 对应子图序号：{subplot_idx}")
        plt.subplot(10, len(labels_names), subplot_idx)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        if count[label] == 0:
            plt.title(label_name)
        count[label] += 1
    plt.show()




if __name__ == '__main__':
    filepath = "./cifar-10-python/cifar-10-batches-py"
    # process_cifar10(filepath)
    data, labels = process_cifar10(filepath)
    # print("data", data)
    img = data.reshape(data.shape[0], 3, 32, 32)
    # print("img:", img)
    labels_names = load_labelsnames(filepath)
    visualable(img, labels, labels_names)
