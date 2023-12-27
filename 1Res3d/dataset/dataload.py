# -*- coding: utf-8 -*-
"""
Load data
==============
**Author**: `zhibin Li`
"""
# Create the dataset  Load sensor data
# 触觉传感器数据读取
####################################################################
import natsort  # 第三方排序库
from PIL import Image
from torchvision.transforms import ToPILImage
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch
import torchvision.transforms as transforms
import os
import numpy as np

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import random

from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision import transforms
import pyqtgraph as pg
import math



class MyDataset(Dataset):  #Dataset
    def __init__(self, path_dir = './data/traindata', transform=None, number_of_segments=2):
        self.number_of_segments = number_of_segments
        self.length_of_segment = math.floor(5000 / self.number_of_segments)

        self.path_dir = path_dir  # file path
        self.transform = transform  # transform
        self.participants = os.listdir(self.path_dir)  # file path list
        self.participants.sort() # Name[]
        self.Gestures_Sensor_root = []  #sensor data list
        self.Gestures_Label = []
        self.Gestures_Vediodata1_root = []  #vedio data

        # data path 和 Label
        #################################
        for name in self.participants:  #name
            #手势
            gesture_dir = self.path_dir + '/' + str(name) + '/1touch'  # sensor data patch

            for gesture_num in range(1,11):
                self.gesture_index = gesture_dir + '/touch' + str(gesture_num)
                for test_num in range(1,5):
                    self.sensor_index = self.gesture_index + '/' + str(test_num) + '/sensor.npz'
                    self.vedio_index = self.gesture_index + '/' + str(test_num) + '/camera1.mp4'
                    #传感器数据路径
                    self.Gestures_Sensor_root.append(self.sensor_index)
                    self.Gestures_Vediodata1_root.append(self.vedio_index)
                    #标签
                    self.Gestures_Label.append(gesture_num-1)  # label




        self.len = len(self.Gestures_Label)  #sensor length




    def __len__(self):  # dataset length
        return self.len * self.number_of_segments


    def judge_where(self, index):
        number_of_index = math.ceil((index + 1) / self.number_of_segments) - 1  # 段数
        number_of_addr = index - number_of_index * self.number_of_segments  #个数
        return number_of_index, number_of_addr


    def __getitem__(self, index):  # index get

        old_index, address_data = self.judge_where(index)

        # path index
        sensor_index_path = self.Gestures_Sensor_root[old_index]  #(./data/traindata/poke\5.npz


        sensor_data = np.load(sensor_index_path)['arr_0'] # sensor data 【5000，8，8】
        sensor_data = sensor_data[self.length_of_segment * address_data: self.length_of_segment * (address_data + 1), :,:]


        sensor_data = torch.from_numpy(sensor_data)
        label = self.Gestures_Label[old_index]

        label = torch.Tensor([label])

        sensor_data = sensor_data.float()  # float set

        return sensor_data,label




if __name__ == '__main__':
    data = MyDataset("/root/data/usr/Zhibin/4Cap_array/I-skin DL project/0Alldata/1Traindata", number_of_segments = 2)

    print(data.__len__())
    sensor, label = data.__getitem__(34) # get the 34th sample
    print(sensor.size()) #torch.Size([5000, 8, 8])
    print(label.size())
