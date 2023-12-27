# -*- coding: utf-8 -*-
"""
Test classification network
Resnet3D_net
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
import dataset.dataload as dataload
import model.Models as Models
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import math

import os
os.chdir("/root/data/usr/Zhibin/4Cap_array/I-skin DL project/1Res3d/")



def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True


def test_acc(number_of_segments = 1):

    setup_seed(9)
    """
    ====================================
    0、Training parameters
    """
    # Number of workers for dataloader
    workers = 4

    batch_size = 60

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 4

    """
    ====================================
    1、Load data
    """
    test_path = r"../0Alldata/3Testdata/"
    testset = dataload.MyDataset(test_path, number_of_segments=number_of_segments)
    print(testset.__len__())

    # Create the test_data
    test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    """
    ====================================
    2、Load model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the generator
    net = Models.Res3D(10, number_of_segments=number_of_segments).to(device)
    net = nn.DataParallel(net, list(range(ngpu)))
    print(net)

    state_dict = torch.load('./result_net/Res3d_s'+ str(number_of_segments) + '.pth')
    net.load_state_dict(state_dict)

    """
    ====================================
    3、Test
    """
    net.eval()
    acc = 0.0
    y_test = []
    y_pred = []
    with torch.no_grad():
        for val_data in test_data:
            val_sensor = val_data[0]
            val_sensor = val_sensor.view(val_sensor.size(0), 1, math.floor(5000 / number_of_segments), 8,
                                         8)  # [10*batch_size,1,50,8,8]

            val_labels = val_data[1]  # label [10]
            val_labels = val_labels.view(val_labels.size(0))  # [10*batch_size]

            val_sensor = val_sensor.to(device)
            val_labels = val_labels.to(device)

            outputs = net(val_sensor)
            predict_y = torch.max(outputs, dim=1)[1]

            # 建立混淆矩阵
            y_test.append(val_labels.cpu().numpy())
            y_pred.append(predict_y.cpu().numpy())

            acc += (predict_y == val_labels).sum().item()

        val_num = len(test_data) * len(val_sensor)

        val_accurate = acc / val_num

        print('testdata_accuracy: %.3f \n' %
              (val_accurate))

    # === 混淆矩阵：真实值与预测值的对比 ===
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(120)
    y_pred = y_pred.reshape(120)
    print(y_test.shape)
    print(y_pred.shape)

    label_name = ["Hit", "Stroke", "Rub", "Pat", "Poke", "Press", "Scratch", "Slap", "Circle", "Put"]
    con_mat = confusion_matrix(y_test, y_pred)

    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)

    # === plot ===
    plt.figure(figsize=(10, 10))

    ax = sns.heatmap(con_mat_norm, annot=True, cmap='OrRd', annot_kws={"fontsize": 15})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    plt.ylim(0, 10)
    # 设置x轴坐标label
    plt.xticks(range(10), label_name, rotation=45, weight='bold')
    # 设置y轴坐标label
    plt.yticks(range(10), label_name, weight='bold')

    plt.xlabel('Predicted labels', size=20, weight='bold')
    plt.ylabel('True labels', size=20, weight='bold')
    plt.savefig('./result_net/test_matrix.png', dpi=600, bbox_inches='tight')

    plt.show()
    print('Finished Test')



if __name__ == '__main__':
    number_of_segments = 1
    test_acc(number_of_segments)


