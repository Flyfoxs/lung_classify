from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
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
from tqdm import tqdm_notebook as tqdm

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
is_cuda = True

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as plt_img
import cv2
from torchvision.datasets.folder import default_loader
from albumentations.pytorch.functional import img_to_tensor


class LungDataset(Dataset):

    def __init__(self, csv_file, root_dir, fold=(0), transform=None, ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file, names=['filename', 'label'])
        if fold is not None:
            self.df.fold = self.df.filename % 5
            self.df = self.df.loc[self.df.fold.isin(fold)]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                f'{self.df.iloc[idx, 0]}.jpg')
        sample = cv2.imread(img_name)

        # sample = sample.transpose((0, 1, 2))

        if self.transform is not None:
            sample = plt_img.fromarray(sample)
            # print(type(sample))
            sample = self.transform(sample)

        # print(type(sample))
        # tensor=img_to_tensor(sample)    # 将图片转化成tensor，

        # print（tensor.shape）  #[3, 224, 224]
        # sample = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
        # sample = Variable(tensor, requires_grad=False)
        label = self.df.iloc[idx, 1]
        return sample, label


transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # transforms.RandomErasing(),
])

train = LungDataset('./input/train.csv', './input/train/', fold=(0, 1, 2, 3), transform=transform)

valid = LungDataset('./input/train.csv', './input/train/', fold=(4,), transform=transform)

# test = ImageFolder('/home/felix/pj/lung_classify/input/')

len(train), len(valid)
# , len(test)


train_data_loader = torch.utils.data.DataLoader(train,batch_size=16,num_workers=3,shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(valid,batch_size=8,num_workers=3,shuffle=True)

from fastai import *
from fastai.vision import *
from fastai.widgets import *
from fastai.vision.learner import cnn_config
from fastai.callbacks.hooks import num_features_model


def create_head(nf: int, ps: Floats = 0.5, concat_pool: bool = True, bn_final: bool = False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)


def create_cnn_model(base_arch, cut: Union[int, Callable] = None, pretrained: bool = True,
                     lin_ftrs: Optional[Collection[int]] = None, ps: Floats = 0.5,
                     custom_head: Optional[nn.Module] = None,
                     bn_final: bool = False, concat_pool: bool = True):
    "Create custom convnet architecture"

    print(type(base_arch), base_arch)
    body = create_body(base_arch, pretrained, cut)

    nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)

    head = create_head(nf, ps=ps, concat_pool=concat_pool, bn_final=bn_final)

    body.requires_grad = False

    model = nn.Sequential(body, head)

    return nn.Sequential(model, nn.Linear(512, 128))


# model_res = create_cnn_model(models.resnet34, cut=None, pretrained=True, lin_ftrs=None, ps=0.5,
#                          custom_head=None,
#         bn_final=False, concat_pool=True)

# model_dense = create_cnn_model( models.densenet161, cut=None, pretrained=True, lin_ftrs=None, ps=0.5,
#                          custom_head=None,
#         bn_final=False, concat_pool=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        from functools import partial
        create_backbone = partial(create_cnn_model, cut=None, pretrained=True, lin_ftrs=None, ps=0.5, custom_head=None, bn_final=False, concat_pool=True)
        self.model_list = [create_backbone(item).cuda() for item in [models.resnet34, models.densenet161, models.vgg19_bn] ]

        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        # x1 = self.model_res(x)
        # x2 = self.model_dense(x)
        # #x3 = self.model_vgg19_bn(x)
        #
        # x = x1 + x2 #+ x3
        x_all = None
        for sigle_model in self.model_list:
            tmp = sigle_model(x)
            x_all = tmp if x_all is None else x_all + tmp

        x = self.fc3(x_all)
        return F.log_softmax(x, dim=1)


Net()



model = Net()
if is_cuda:
    model.cuda()


optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(data_loader), desc=f'{epoch}:{phase}', total=len(data_loader)):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        #         print(output.shape, target.shape)
        loss = F.nll_loss(output, target)
        #         print(loss, loss.shape)
        #         print(F.nll_loss(output,target,size_average=False).data.shape)

        #         loss_val = F.nll_loss(output,target,size_average=False)
        #         total_loss += loss_val.data
        running_loss += F.nll_loss(output, target, size_average=False).data
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')

    return loss, accuracy


train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]

patient = 3
best_epoch = 0
for epoch in range(1,30):
    epoch_loss, epoch_accuracy = fit(epoch,model,train_data_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,valid_data_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    best_score = np.max(val_accuracy)
    if val_epoch_accuracy == best_score:
        best_epoch = epoch
        wait = 0
    else:
        wait = wait + 1

    if wait >= patient:
        print(f'Early stop wait {wait} epochs, and current epoch:{epoch}, best:{best_score:6.5f}/{best_epoch:02} ')


"""" 
python -u customer/classify_ensemble.py main with conf_name=5cls_resnet34  fold=0
"""