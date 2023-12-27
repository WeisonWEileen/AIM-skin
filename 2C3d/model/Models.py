'''
Spatiotemporal Touch Perception network
Res3D C3D C2D
==============
**Author**: `zhibin Li`__
'''

import torch.nn as nn
import torch.nn.init as init
import torch
import math

class ResBlock(nn.Module):
    def __init__(self, in_channel,out_channel, spatial_stride=1,temporal_stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, out_channel,kernel_size=(3,3,3),stride=(temporal_stride,spatial_stride,spatial_stride),padding=(1,1,1))
        self.conv2 = nn.Conv3d(out_channel, out_channel,kernel_size=(3, 3, 3),stride=(1, 1, 1),padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        if in_channel != out_channel or spatial_stride != 1 or temporal_stride != 1:
            self.down_sample=nn.Sequential(nn.Conv3d(in_channel, out_channel,kernel_size=1,stride=(temporal_stride,spatial_stride,spatial_stride),bias=False),
                                           nn.BatchNorm3d(out_channel))
        else:
            self.down_sample=None

    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2(x_branch)
        x_branch = self.bn2(x_branch)
        if self.down_sample is not None:
            x=self.down_sample(x)
        return self.relu(x_branch+x)

class Res3D(nn.Module):
    # Input size: 8x224x224
    def __init__(self, num_class,number_of_segments = 1):
        super(Res3D, self).__init__()

        self.conv11 = nn.Conv3d(1,32,kernel_size=(20,2,2),stride=(10,1,1),padding=(5,0,0))
        self.conv12 = nn.Conv3d(32, 64, kernel_size=(20, 2, 2), stride=(10, 1, 1), padding=(5, 1, 1))

        #(10,64,50, 8,8)
        self.conv2  = nn.Sequential(ResBlock(64,64,spatial_stride=1,temporal_stride=1),
                                 ResBlock(64, 64))
        self.conv3 = nn.Sequential(ResBlock(64,128,spatial_stride=1,temporal_stride=2),
                                 ResBlock(128, 128))
        self.conv4 = nn.Sequential(ResBlock(128, 256, spatial_stride=1,temporal_stride=2),
                                   ResBlock(256, 256))
        self.conv5 = nn.Sequential(ResBlock(256, 512, spatial_stride=2,temporal_stride=2),
                                   ResBlock(512, 512))
        avg_size = math.floor(7/number_of_segments)
        if(number_of_segments in [4, 5]): avg_size = 2
        if (avg_size<1): avg_size = 1
        self.avg_pool=nn.AvgPool3d(kernel_size=(avg_size,4,4))
        self.linear=nn.Linear(512,num_class)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        x = self.conv11(x)   #
        x = self.conv12(x)  # [10, 64, 50, 8, 8]

        x=self.conv2(x)   #[10, 64, 50, 8, 8]
        x=self.conv3(x)   #[10, 128, 25, 8, 8]
        x=self.conv4(x)   #[10, 256, 13, 8, 8]
        x = self.conv5(x) #[10, 512, 7, 4, 4]

        x=self.avg_pool(x) #[10, 512, 1, 1, 1]
        x = self.linear(x.view(x.size(0),-1)) #[10,10]
        x = self.softmax(x)
        #print(x.size())
        return x

class C3D(nn.Module):

    def __init__(self):
        super(C3D, self).__init__()


        self.conv1 = nn.Conv3d(1, 64, kernel_size=(10, 2, 2), )
        self.pool1 = nn.MaxPool3d(kernel_size=(10, 2, 2), stride=(15, 1, 1))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(10, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(15, 1, 1))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(4, 3, 3), padding=(0, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(4, 3, 3), padding=(0, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(15, 2, 2))
        #

        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim =1)

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)


        #
        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)


        h = h.view(-1, 256)
        h = self.relu(self.fc4(h))
        h = self.dropout(h)
        h = self.relu(self.fc5(h))
        h = self.dropout(h)
        #
        h = self.fc6(h)
        h = self.softmax(h)

        return h


class C2D(nn.Module):

    def __init__(self):
        super(C2D, self).__init__()


        self.conv1 = nn.Conv2d(1, 64, kernel_size=(10, 2), )
        self.pool1 = nn.MaxPool2d(kernel_size=(15, 2), stride=(15, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(10, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(15, 4), stride=(15, 4))
        #
        self.conv3a = nn.Conv2d(128, 256, kernel_size=(4, 3), padding=(0, 1))
        self.conv3b = nn.Conv2d(256, 256, kernel_size=(4, 3), padding=(0, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(10, 4), stride=(10, 4))

        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim =1)

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)           #[10, 64, 499, 14]

        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        #
        #
        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        #
        #
        h = h.view(-1, 256)
        h = self.relu(self.fc4(h))
        h = self.dropout(h)
        h = self.relu(self.fc5(h))
        h = self.dropout(h)
        #
        h = self.fc6(h)
        h = self.softmax(h)

        return h



if __name__ == '__main__':
    net = Res3D(10)
    print(net)

    input = torch.rand(10,1,5000, 8,8)  #(10,3,8, 224,224) >>(Batch,C通道,L帧长,H高，w宽)

    out = net(input)

    print(out.size())