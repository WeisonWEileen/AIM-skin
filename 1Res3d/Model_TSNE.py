import dataset.dataload as dataload
import model.Models as Models

import torch
import torch.nn as nn
import torchvision

from collections import OrderedDict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random

import os
os.chdir("/root/data/usr/Zhibin/4Cap_array/I-skin DL project/1Res3d/")

number_of_segments = 1


def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True



def get_data():
    """
        ====================================
            1、load model
        """
    #device
    device = torch.device('cpu')
    net = Models.Res3D(10,number_of_segments=number_of_segments).to(device)

    state_dict = torch.load('./result_net/Res3d_s' + str(number_of_segments) + '.pth', map_location=device)

    # more GPU
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # 去掉 `module.`
        # name = k.replace(“module.", "")
        state_dict_new[name] = v

    net.load_state_dict(state_dict_new)
    print(net)

    """
    ====================================
        2、Feature extraction
    """

    return_layers = {'avg_pool': 'feature_512'}
    backbone = torchvision.models._utils.IntermediateLayerGetter(net, return_layers)

    backbone.eval()
    x = torch.randn(10, 1, 5000, 8, 8)
    out = backbone(x)
    print(out['feature_512'].shape)


    """
    ====================================
        3、Load data
    """
    train_path = r"../0Alldata/1Traindata/"
    validation_path = r"../0Alldata/2Validationdata/"
    test_path = r"../0Alldata/3Testdata/"

    trainset = dataload.MyDataset(train_path,number_of_segments=number_of_segments)
    validationset = dataload.MyDataset(validation_path, number_of_segments=number_of_segments )
    testset = dataload.MyDataset(test_path, number_of_segments=number_of_segments)

    # Create the test_data
    train_data = torch.utils.data.DataLoader(trainset, batch_size=10,
                                             shuffle=True, num_workers=1)
    validation_data = torch.utils.data.DataLoader(validationset, batch_size=10,
                                             shuffle=True, num_workers=1)
    test_data = torch.utils.data.DataLoader(testset, batch_size=10,
                                             shuffle=True, num_workers=1)

    data_state = 0

    #1Traindata
    for step, data in enumerate(train_data, start=0):
        sensor = data[0]
        sensor = sensor.view(sensor.size(0), 1, 5000, 8, 8)  # [10*batch_size,1,50,8,8]
        labels = data[1]  # label [10]
        labels = labels.view(labels.size(0))  # [10*batch_size]

        out = backbone(sensor)
        feature_512 = out['feature_512']

        if (data_state == 0):
            outdata = feature_512
            out_label = labels
            data_state = 1
        else:
            outdata = torch.cat((outdata,feature_512), 0)
            out_label = torch.cat((out_label, labels), 0)

    #2Validationdata
    for step, data in enumerate(validation_data, start=0):
        sensor = data[0]
        sensor = sensor.view(sensor.size(0), 1, 5000, 8, 8)  # [10*batch_size,1,50,8,8]
        labels = data[1]  # label [10]
        labels = labels.view(labels.size(0)) # [10*batch_size]

        out = backbone(sensor)
        feature_512 = out['feature_512']

        outdata = torch.cat((outdata,feature_512), 0)
        out_label = torch.cat((out_label, labels), 0)

    #3Testdata
    for step, data in enumerate(test_data, start=0):
        sensor = data[0]
        sensor = sensor.view(sensor.size(0), 1, 5000, 8, 8)
        labels = data[1]
        labels = labels.view(labels.size(0))

        out = backbone(sensor)
        feature_512 = out['feature_512']

        outdata = torch.cat((outdata,feature_512), 0)
        out_label = torch.cat((out_label, labels), 0)


    outdata = outdata.view(outdata.size(0),-1)
    outdata = outdata.detach().numpy()
    out_label = out_label.detach().numpy()

    return outdata, out_label




if __name__ == '__main__':
    setup_seed(9)
    datas , labels = get_data()


    tsne = TSNE(n_components=2, perplexity = 10, init='pca', random_state=9)
    result = tsne.fit_transform(datas,labels)
    ax = plt.subplot()
    ax.set_title('2D t-SNE of tactile features')
    scatter = ax.scatter(result[:, 0], result[:, 1], c=labels)
    legendClass = ax.legend(*scatter.legend_elements(prop="colors"),
                            loc="upper left", title="classes")
    ax.add_artist(legendClass)
    plt.colorbar(scatter)
    plt.savefig('./result_net/tsne.png', dpi=600,bbox_inches='tight')
    plt.show()
