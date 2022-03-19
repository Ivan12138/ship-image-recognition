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


def visualable():
    figure = plt.figure(figsize=len())


if __name__ == '__main__':
    filepath = "./cifar-10-python/cifar-10-batches-py"
    # process_cifar10(filepath)
    load_labelsnames(filepath)

