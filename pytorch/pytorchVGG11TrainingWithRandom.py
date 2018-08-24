from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch.optim as optim
from os import listdir
from os.path import isfile, join
import matplotlib.mlab as mlab
import pandas as pd
import random


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=3, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    kernel_size = [(3, 7), (3, 5), (2, 4), (2, 3), (2, 2)]
    layers = []
    in_channels = 18
    compt=0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            compt+=1
        else:
            conv2d = nn.Conv2d(in_channels, v,stride=(1,2), kernel_size=kernel_size[compt], padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


# read data from binary
def data_generator(training=True):

    d = np.fromfile('no3.bin', dtype='<i2', count=-1)
    N = 32
    d = d.reshape((N, -1), order='F')

    bat1 = pd.read_csv('./batxyz2.csv')

    while(True):
        batch_count=0
        batch_spec=[]
        batch_pos=[]
        if training:
            start_point=1606200
        else:
            start_point=1606200+(15000*2397)

        for current_loc in np.arange(start_point, 46546200, 1500):
            #    print('current_loc:', current_loc)

            #
            # bat1 xyz coordinates
            #
            current_time = current_loc * 2e-6  # 2 micro sec = 1 sample of 500kHz
            NNtime = (bat1.time - current_time).abs().idxmin()
            NNtime_pm1 = bat1.loc[NNtime - 1:NNtime + 1]  # NNtime-1 and NNtime

            #
            # Linear interpolation
            #
            bat1x = np.interp(current_time, NNtime_pm1.time, NNtime_pm1['bat(X)'])
            bat1y = np.interp(current_time, NNtime_pm1.time, NNtime_pm1['bat(Y)'])
            bat1z = np.interp(current_time, NNtime_pm1.time, NNtime_pm1['bat(Z)'])

            #
            # specgram
            #
            step = int((1 / 30.) / (1 / 500000.))  # 30fps 1frame vs 500kHz sampling
            step  # 1frame = 16666 sample points
            st = int(current_loc - step / 2)
            et = int(current_loc + step / 2)
            B_all = []
            for i in range(22):  # use ch.0 -- ch.21
                if i in [0, 10, 20, 21]:
                    continue

                try:
                    time_laps = d[i, st:et]
                    # add random at maximum two percent
                    time_laps = time_laps + time_laps * (random.randint(-2, 2)/100)
                    B, F, T = mlab.specgram(time_laps,
                                            NFFT=128,
                                            Fs=500000,  # 500kHz
                                            window=mlab.window_hanning,
                                            noverlap=126
                                            )

                    # get B[2:34, :] --> [32, 8270]
                    B = B[2:34, :]

                    B_all.append(B)
                except:
                    pass
            B_all = np.dstack(B_all)  # 3D array
            B_all /= 40000  # ad-hoc normalizatoin
            # print('current_loc:', current_loc, [B_all.max(), B_all.min()], [bat1x, bat1y, bat1z])

            batch_spec.append(B_all)
            batch_pos.append(np.array([bat1x, bat1y, bat1z]))

            if batch_count==9:
                batch_count=0
                yield_spec=batch_spec
                yield_pos=batch_pos
                batch_pos=[]
                batch_spec=[]
                yield (np.array(yield_spec),np.array(yield_pos))
            else:
                batch_count+=1

# def data_generator(datalist):
#     data = np.load(datalist)
#     batch_size = 10
#     step = len(data) // batch_size
#
#     while (True):
#         # print("Epoch finish")
#         np.random.shuffle(data)
#         for i in range(step):
#             X = []
#             Y = []
#             for j in range(batch_size):
#                 spec = np.load("data/specgram_" + data[i * batch_size + j] + ".npy")
#                 true_xyz = np.load("data/trueXYZ_" + data[i * batch_size + j] + ".npy")
#                 X.append(spec)
#                 Y.append(true_xyz)
#
#             # print("X : ", len(X))
#             X = np.array(X)
#             Y = np.array(Y)
#
#             yield (X, Y)

def create_two_list(folder):
    only_files = [f[9:-4] for f in listdir(folder) if (isfile(join(folder, f)) and "specgram_" in f)]

    test_g = []
    train_g =[]
    divide = len(only_files) * 0.2
    k = 0
    for name in only_files:
        if k<divide :
            test_g.append(name)
        else :
            train_g.append(name)
        k=k+1
    np.save("./newtrainlist.npy",train_g)
    np.save("./newtestlist.npy",test_g)

#create to list file with trainning and testing data name
# create_two_list("./data")

#create generator
train_gen = data_generator(True)

#create model
vgg_var = vgg11_bn()
vgg_var.cuda()
vgg_var.load_state_dict(torch.load("./logsVGG11Random/saveVGG11Random.pth"))

#show summary
print(vgg_var)

#set model to train
vgg_var.train()

#prepare optimizer
optimizer = optim.Adam(vgg_var.parameters(), lr=0.01)

#for 100 epochs
for e in range(100) :
    i=0
    all_loss_epoch = []
    print("Epoch : {0}".format(e))
    for d in train_gen:
        #reset grad to avoid accumulation
        optimizer.zero_grad()

        #predict
        data_tmp=np.einsum('klij->kjli', d[0])
        data = torch.from_numpy(data_tmp)
        data = data.type(torch.cuda.FloatTensor)

        # print(data_tmp.shape)
        prediction = vgg_var(data)

        true_coordinate = torch.from_numpy(d[1])
        true_coordinate = true_coordinate.type(torch.cuda.FloatTensor)

        # prepare loss mean square error
        lossMSE = nn.MSELoss()

        #calculate loss
        loss = lossMSE(prediction, true_coordinate)
        all_loss_epoch.append(loss.data)
        print("Step : {0}, loss : {1}".format(i,loss))

        loss.backward()

        #update weight
        optimizer.step()

        i+=1
        if i > 2395:
            break
    nparray = np.array(all_loss_epoch).flatten()
    print("Mean loss of epoch {0} : {1}".format(e, nparray.mean()))
    # save weight
    torch.save(vgg_var.state_dict(), "./logsVGG11Random/pytorch{0}.pth".format(e))

# load model weight
# the_model = vgg13_bn()
# the_model.load_state_dict(torch.load("./logs/pytorch10"))
#

