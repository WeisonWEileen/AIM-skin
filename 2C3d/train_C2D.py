# -*- coding: utf-8 -*-
"""
Training classification network
C2D
==============
**Author**: `zhibin Li`__
"""
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import time
import dataset.dataload as dataload #加载数据集
import model.Models as Models #加载数据集
import random
import math
import argparse

import os
os.chdir("/root/data/usr/Zhibin/4Cap_array/I-skin DL project/2C3d")



parser = argparse.ArgumentParser()
parser.add_argument('--number_of_segments', type=int, default=1)
args = parser.parse_args()


number_of_segments = args.number_of_segments  # Shear segment number

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    setup_seed(9)

    """
    ====================================
    0、Training parameters
    """
    # Number of workers for dataloader
    workers = 4  # 4

    # Batch size during training
    batch_size = 60  # 10

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 4

    # Number of training epochs
    num_epochs = 50

    # Learning rate for optimizers
    lr = 0.0001

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    """
    ====================================
    1、Load data
    """

    train_path = r"../0Alldata/1Traindata/"
    validation_path = r"../0Alldata/2Validationdata/"




    trainset = dataload.MyDataset(train_path,number_of_segments = number_of_segments)
    validationset = dataload.MyDataset(validation_path,number_of_segments = number_of_segments)

    print(trainset.__len__())


    # Create the test_data
    train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Create the test_data
    validation_data = torch.utils.data.DataLoader(validationset, batch_size=10,
                                             shuffle=True, num_workers=workers)



    """
    ====================================
    2、Load model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the generator
    #net = Models.Res3D(10, number_of_segments=number_of_segments).to(device)
    net = Models.C2D().to(device)
    net = nn.DataParallel(net, list(range(ngpu)))
    print(net)


    """
    ====================================
    3、Initial set
    """

    # Loss Functions and Optimizers
    loss_function = nn.CrossEntropyLoss()

    # Setup Adam optimizers for both G and D
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=0.01)

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    save_path = './result_net/C2d_s'+ str(number_of_segments) + '.pth'
    best_acc = 0.0
    Accuracy = []



    """
    ====================================
    4、Train
    """

    for epoch in range(num_epochs):
        ########################################## train ###############################################
        net.train()
        running_loss = 0.0
        time_start = time.perf_counter()

        if(epoch==20):
            # Setup Adam optimizers for both G and D
            lr = lr*0.1
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)  # 优化器
        if(epoch==40):
            # Setup Adam optimizers for both G and D
            lr = lr*0.1
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)  # 优化器


        for step, data in enumerate(train_data, start=0):

            sensor = data[0]
            sensor = sensor.view(sensor.size(0),1,math.floor(5000 / number_of_segments),64)   # [10*batch_size,1,50,8,8]


            labels = data[1]      #label [10]
            labels = labels.view( labels.size(0))       # [10*batch_size]



            sensor = sensor.to(device,dtype=torch.float)
            labels = labels.to(device)

            optimizer.zero_grad()  # 清除历史梯度
            outputs = net(sensor)  # 正向传播



            loss = loss_function(outputs,labels.long())
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数
            running_loss += loss.item()

            # 打印训练进度（使训练过程可视化）
            rate = (step + 1) / len(train_data)  # 当前进度 = 当前step / 训练一轮epoch所需总step
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print('%f s' % (time.perf_counter() - time_start))

        ########################################### validate ###########################################
        net.eval()  # 验证过程中关闭 Dropout
        acc = 0.0
        with torch.no_grad():
            for val_data in validation_data:
                val_sensor = val_data[0]  # 获取数据 [10,1,50,8,8]
                val_sensor = val_sensor.view(val_sensor.size(0),1,math.floor(5000 / number_of_segments),64)   # [10*batch_size,1,50,8,8]

                val_labels = val_data[1]  # label [10]
                val_labels = val_labels.view( val_labels.size(0))  # [10*batch_size]

                val_sensor = val_sensor.to(device)  # 转换统一数据格式
                val_labels = val_labels.to(device)

                outputs = net(val_sensor)
                predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
                acc += (predict_y == val_labels).sum().item()
            # print('predict_y',predict_y)   #显示结果验证
            # print('val_labels', val_labels.to(device))
            val_num = len(validation_data) * len(val_sensor)  #测试集总数


            val_accurate = acc / val_num
            Accuracy.append(val_accurate)

            #保存准确率最高的那次网络参数
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

            print('[epoch %d] train_loss: %.3f  validation_accuracy: %.3f \n' %
                  (epoch + 1, running_loss / step, val_accurate))

    print('Finished Training')



