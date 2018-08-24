from __future__ import print_function
import torchvision
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
import torch.optim as optim
from os import listdir
from os.path import isfile, join

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
    kernel_size = [(3, 7), (3, 5), (3, 4), (3, 3), (3, 3)]
    layers = []
    in_channels = 18
    compt=0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2,padding=1)]
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

def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def data_generator(datalist):
    data = np.load(datalist)
    batch_size = 10
    step = len(data) // batch_size

    while (True):
        # print("Epoch finish")
        np.random.shuffle(data)
        for i in range(step):
            X = []
            Y = []
            for j in range(batch_size):
                spec = np.load("data/specgram_" + data[i * batch_size + j] + ".npy")
                true_xyz = np.load("data/trueXYZ_" + data[i * batch_size + j] + ".npy")
                X.append(spec)
                Y.append(true_xyz)

            # print("X : ", len(X))
            X = np.array(X)
            Y = np.array(Y)

            yield (X, Y)

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
train_gen = data_generator("./newtestlist.npy")

#create model
vgg_var = vgg16_bn()
vgg_var.load_state_dict(torch.load("./logs/save1"))
vgg_var.cuda()

#show summary
print(vgg_var)

#set model to train
vgg_var.eval()

#prepare optimizer
optimizer = optim.Adam(vgg_var.parameters(), lr=0.01)

i=0
all_loss_epoch = []
for d in train_gen:
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

    i+=1
    if i > 598:
        break

nparray = np.array(all_loss_epoch).flatten()
print("Mean test loss : {0}".format(nparray.mean()))
# save weight

# load model weight
# the_model = vgg13_bn()
# the_model.load_state_dict(torch.load("./logs/pytorch10"))
#

