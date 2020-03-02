from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

from efficientnet_pytorch import EfficientNet
from fastai import *
from fastai.vision import *
from fastai.widgets import *
from fastai.vision.learner import cnn_config
from fastai.callbacks.hooks import num_features_model


from task_distribute.locker import task_locker
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
import time
from tqdm import tqdm

import os
def create_head(nf: int, nc:int, ps: Floats = 0.5, concat_pool: bool = True, bn_final: bool = False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if nc >0 else [nf, 512]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)


def create_cnn_model(base_arch, nc:int, cut: Union[int, Callable] = None, pretrained: bool = True,
                     lin_ftrs: Optional[Collection[int]] = None, ps: Floats = 0.5,
                     custom_head: Optional[nn.Module] = None,
                     bn_final: bool = False, concat_pool: bool = True):
    "Create custom convnet architecture"

    print(type(base_arch), base_arch)
    body = create_body(base_arch, pretrained, cut)
    # body.requires_grad = True

    nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)

    head = create_head(nf, nc,  ps=ps, concat_pool=concat_pool, bn_final=bn_final)

    model = nn.Sequential(body, head)



    return model


# model_res = create_cnn_model(models.resnet34, cut=None, pretrained=True, lin_ftrs=None, ps=0.5,
#                          custom_head=None,
#         bn_final=False, concat_pool=True)

# model_dense = create_cnn_model( models.densenet161, cut=None, pretrained=True, lin_ftrs=None, ps=0.5,
#                          custom_head=None,
#         bn_final=False, concat_pool=True)

class Net(nn.Module):
    def __init__(self, subnet_list, nc):
        super().__init__()

        from functools import partial
        create_backbone = partial(create_cnn_model, nc=128, cut=None, pretrained=True, lin_ftrs=None, ps=0.5, custom_head=None, bn_final=False, concat_pool=True)
        #self.create_backbone(item).cuda() for item in [models.resnet34, models.densenet161, models.vgg19_bn] ]

        if isinstance(subnet_list, str) : subnet_list = subnet_list.split(',')
        subnet_list = [item.strip() for item in subnet_list]

        print('subnet_list=', subnet_list)

        for network_name in subnet_list:
            import torchvision.models as to_models
            backbone_fn = getattr(to_models, network_name)
            backbone_tmp = create_backbone(backbone_fn)
            setattr(self, f'ens_{network_name}', backbone_tmp)

        self.fc3 = nn.Linear(128, nc)

    def forward(self, x):
        # x1 = self.model_res(x)
        # x2 = self.model_dense(x)
        # #x3 = self.model_vgg19_bn(x)
        #
        # x = x1 + x2 #+ x3

        #print('type', type(list(self._modules.values())[0]))
        #print('self._modules.values', list(self._modules.values())[0])
        #x_all = functools.reduce(lambda a, b: a(x) + b(x), list(self._modules.values())[:-1])

        if len(list(self._modules.values())) == 1:
            return list(self._modules.values())[-1](x)

        #print('==='*10, len(list(self._modules.values())))
        x_all = torch.zeros(1).cuda()
        for sub_model in list(self._modules.values())[:-1]:
            x_all = x_all + sub_model(x)
        #x_all = self.model_res(x) + self.model_dense(x) + self.model_vgg19_bn(x) +

        # mod_list = list(self._modules.values())[:-1]
        # x_all = torch.cat([item(x) for item in mod_list], -1)
        return self.fc3(x_all)


def get_backbone(backbone_name):
    if backbone_name == 'resnet34':
        backbone = models.resnet34
    elif backbone_name == 'resnet50':
        backbone = models.resnet50
    elif backbone_name == 'densenet161':
        backbone = models.densenet161
    elif backbone_name == 'densenet201':
        backbone = models.densenet201
    else:
        raise Exception('No model')

    return backbone

def weights_init(m):
    import torch.nn.init.xavier_uniform as xavier
    if isinstance(m, nn.Linear):
        xavier(m.weight.data)
        xavier(m.bias.data)

def get_network(backbone_conf, nc):
    if ',' in backbone_conf:
        backbone_conf = backbone_conf.split(',')

    model = Net(backbone_conf, nc).cuda()
    # flatten_model = lambda m: sum(map(flatten_model, children_and_parameters(m)), []) if num_children(m) else [m]
    # model_list = flatten_model(model)
    # freeze_to = -3
    # print(f'Lock to {freeze_to} layers, layer need to learn:====')
    # for layer in model_list[freeze_to:]:
    #     print(f'Layer need to learn: {layer}')
    #     cond_init(layer, nn.init.kaiming_normal_)
    #     layer.requires_grad = True
    # for layer in model_list[:freeze_to]:
    #     layer.requires_grad = False
    # # print(model)
    return model




