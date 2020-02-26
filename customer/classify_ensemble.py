from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
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


is_cuda = True
version = 'en02'

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as plt_img
import cv2
from torchvision.datasets.folder import default_loader
from albumentations.pytorch.functional import img_to_tensor
from sacred import Experiment
from easydict import EasyDict as edict
from sacred.observers import MongoObserver

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

def get_dl(valid_fold,image_size):
    valid_fold = int(valid_fold)
    transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.RandomErasing(),
    ])

    train_folds = list(range(5))
    train_folds.remove(valid_fold)
    train = LungDataset('./input/train.csv', './input/train/', fold=train_folds, transform=transform)

    valid = LungDataset('./input/train.csv', './input/train/', fold=(valid_fold,), transform=transform)

    # test = ImageFolder('/home/felix/pj/lung_classify/input/')

    print(len(train), len(valid))
    # , len(test)

    train_data_loader = torch.utils.data.DataLoader(train,batch_size=16,num_workers=3,shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(valid,batch_size=8,num_workers=3,shuffle=True)

    return train_data_loader, valid_data_loader



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
    def __init__(self, subnet_list):
        super().__init__()

        from functools import partial
        create_backbone = partial(create_cnn_model, cut=None, pretrained=True, lin_ftrs=None, ps=0.5, custom_head=None, bn_final=False, concat_pool=True)
        #self.create_backbone(item).cuda() for item in [models.resnet34, models.densenet161, models.vgg19_bn] ]

        if isinstance(subnet_list, str) : subnet_list = subnet_list.split(',')

        if 'resnet34' in subnet_list: self.model_res = create_backbone(models.resnet34)

        if 'densenet161' in subnet_list: self.model_dense = create_backbone(models.densenet161)

        if 'vgg19_bn' in subnet_list: self.model_vgg19_bn = create_backbone(models.vgg19_bn)

        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        # x1 = self.model_res(x)
        # x2 = self.model_dense(x)
        # #x3 = self.model_vgg19_bn(x)
        #
        # x = x1 + x2 #+ x3

        #print('type', type(list(self._modules.values())[0]))
        #print('self._modules.values', list(self._modules.values())[0])
        #x_all = functools.reduce(lambda a, b: a(x) + b(x), list(self._modules.values())[:-1])

        x_all = torch.zeros(1).cuda()
        for sub_model in list(self._modules.values())[:-1]:
            x_all = x_all + sub_model(x)
        #x_all = self.model_res(x) + self.model_dense(x) + self.model_vgg19_bn(x) +

        x = self.fc3(x_all)
        return F.log_softmax(x, dim=1)


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
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

    loss = float(running_loss / len(data_loader.dataset))
    accuracy = float(100. * running_correct / len(data_loader.dataset))

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')

    return loss, accuracy





###### sacred begin
from sacred import Experiment
from easydict import EasyDict as edict
from sacred.observers import MongoObserver
from sacred import SETTINGS

ex = Experiment('lung')
db_url = 'mongodb://sample:password@10.10.20.103:27017/db?authSource=admin'
ex.observers.append(MongoObserver(url=db_url, db_name='db'))
#SETTINGS.CAPTURE_MODE = 'sys'

@ex.config
def my_config():
    conf_name = None
    fold = -1


# @ex.command()
# def main(_config):
#
#     config = edict(_config)
#     conf_name = config.conf_name
#     valid_fold = int(config.fold)
#
#     train(valid_fold, conf_name)


@ex.command()
def main(_config):
    config = edict(_config)

    file_name = config.conf_name
    if not os.path.exists(file_name):
        print(f'Can not find file:{file_name}')
        file_name = f'./configs/{file_name}.yaml'
    f = open(file_name)
    conf = edict(yaml.load(f))

    model = Net(conf.backbone)
    valid_fold = int(config.fold)


    model.cuda()

    print(model)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []

    patient = 3
    best_epoch = 0

    train_data_loader, valid_data_loader = get_dl(valid_fold, conf.image_size)
    print(conf)
    for epoch in tqdm(range(1, 30), desc='epoch'):
        epoch_loss, epoch_accuracy = fit(epoch, model, train_data_loader, phase='training')
        ex.log_scalar('train.acc', epoch_accuracy, step=epoch)
        ex.log_scalar('train.loss', epoch_loss, step=epoch)

        val_epoch_loss, val_epoch_accuracy = fit(epoch, model, valid_data_loader, phase='validation')
        ex.log_scalar('val.loss', val_epoch_loss, step=epoch)
        ex.log_scalar('val.acc', val_epoch_accuracy, step=epoch)

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
            print(f'Early stop wait {wait} epochs, and current epoch:{epoch}, fold:{valid_fold}, best:{best_score:6.5f}/{best_epoch:02}, {file_name} ')
            break

    print('epoch_accuracy', epoch_accuracy )
    print('val_epoch_accuracy', val_epoch_accuracy )

if __name__ == '__main__':

    """" 
    python -u customer/classify_ensemble.py main with conf_name=ens_res_den_vgg  fold=0
    """

    from sacred.arg_parser import get_config_updates
    import sys
    config_updates, named_configs = get_config_updates(sys.argv[1:])
    conf_name = config_updates.get('conf_name')
    fold = config_updates.get('fold')


    locker = task_locker('mongodb://sample:password@10.10.20.103:27017/db?authSource=admin', remove_failed =9 , version=version)
    task_id = f'lung_{conf_name}_{fold}'
    #pydevd_pycharm.settrace('192.168.1.101', port=1234, stdoutToServer=True, stderrToServer=True)
    with locker.lock_block(task_id=task_id) as lock_id:
        if lock_id is not None:
            ex.add_config({
                'lock_id': lock_id,
                'lock_name': task_id,
                'version': version,
            })
            res = ex.run_commandline()


