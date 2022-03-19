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
def loadcifar(filename):
    with open(filename, "rb") as f:
        batch4 = pickle.load(f, encoding='bytes')
    # [b'batch_label', b'labels', b'data', b'filenames']
    print(batch4.keys())
    data, labels = batch4[b'data'], batch4[b'labels']
    return data, labels

# load label_names
def load_labelsnames(filepath):
    filename = os.path.join(filepath, 'batches.meta')
    with open(filename, 'rb') as f:
        batches_meta = pickle.load(f, encoding='bytes')
    # [b'num_cases_per_batch', b'label_names', b'num_vis']
    print(batches_meta.keys())
    labels_names = batches_meta[b'label_names']
    return labels_names


# 处理数据集
def process_cifar10(filepath):
    filename = os.path.join(filepath, "data_batch_4")
    data, labels = loadcifar(filename)
    return data, labels



# def topil(data)
#


def visualable(img, labels, labels_names):
    figure = plt.figure(figsize=(len(labels_names), 10))

    idxs = list(range(len(img)))
    np.random.shuffle(idxs)
    count = [0]*len(labels_names)
    for idx in idxs:
        label = labels[idx]
        # ???
        if count[label]>=10:
            continue
        if sum(count)>10*len(labels_names):
            pass


if __name__ == '__main__':
    filepath = "./cifar-10-python/cifar-10-batches-py"
    # process_cifar10(filepath)
    data, labels = process_cifar10(filepath)
    print("data", data)
    img = data.reshape(data.shape[0], 3, 32, 32)
    print("img:", img)
    labels_names = load_labelsnames(filepath)
    # visualable(img, labels, labels_names)
