import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import os
from task_distribute.locker import task_locker
from file_cache.cache import file_cache, logger, timed

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

from tqdm import tqdm, tqdm_notebook
from glob import glob
import itertools
import numpy as np
from easydict import EasyDict as edict

import warnings
warnings.filterwarnings("ignore")
import time
import pandas as pd


import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision.datasets.folder import default_loader
from albumentations.pytorch.functional import img_to_tensor

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
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
            sample = Image.fromarray(sample)
            # print(type(sample))
            sample = self.transform(sample)

        # print(type(sample))
        # tensor=img_to_tensor(sample)    # 将图片转化成tensor，

        # print（tensor.shape）  #[3, 224, 224]
        # sample = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
        # sample = Variable(tensor, requires_grad=False)
        label = self.df.iloc[idx, 1]
        return sample, label


from fastai import *
from fastai.vision import *
from fastai.widgets import *

from fastai.callbacks import EarlyStoppingCallback,SaveModelCallback


from easydict import EasyDict as edict
import yaml, os


from sacred import Experiment
from easydict import EasyDict as edict
from sacred.observers import MongoObserver



version = 'ctm15'
is_cuda = True

def get_oof_df(learn, ds_type ):
    res = learn.get_preds(ds_type=ds_type)
    res = res[0].numpy()
    label = None
    if ds_type == DatasetType.Test:
        index = learn.data.test_ds.items
    elif ds_type == DatasetType.Valid:
        index = learn.data.valid_ds.items

        label = [item.data for item in learn.data.valid_ds.y]
    else:
        print(ds_type)
    index = [int(os.path.basename(file).split('.')[0])for file in list(index)]
    df = pd.DataFrame(res, index=index)
    if label is not None:
        df['label'] = label
    return df


def save_stack_feature(train: pd.DataFrame, test: pd.DataFrame, file_path):
    train.to_hdf(file_path, 'train', mode='a')
    test.to_hdf(file_path, 'test', mode='a')
    logger.info(f'OOF file save to :{file_path}')
    return train, test


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0

    optimizer = optim.SGD(model.classifier.parameters(), lr = 0.0001, momentum = 0.5)

    for batch_idx, (data, target) in tqdm(enumerate(data_loader), desc=f'{phase},{epoch:02}', total=len(data_loader)):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        running_loss += F.cross_entropy(output, target, size_average=False).data  # [0]
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


def train(valid_fold , conf_name):

    f = open(f'./configs/{conf_name}.yaml')
    conf = edict(yaml.load(f))

    class_cnt = conf.class_cnt
    backbone_name = conf.backbone
    unfreeze = True #conf.unfreeze if 'unfreeze' in conf else False
    epoch = 50

    assert int(valid_fold) <= 4
    # batch_id = str(round(time.time()))
    backbone = get_backbone(backbone_name)

    class_cnt = 4
    model = create_cnn_model(models.resnet34, class_cnt, cut=None, pretrained=True, lin_ftrs=None, ps=0.5,
                             custom_head=None,
                             bn_final=False, concat_pool=True)

    print(model)

    backbone.classifier[-1].out_features = 5
    for param in model.features.parameters(): param.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    ])

    train_folds = range(5)
    train_folds.remove(4)
    train = LungDataset('../input/train.csv', '../input/train/', fold=train_folds, transform=transform)

    valid = LungDataset('../input/train.csv', '../input/train/', fold=(valid_fold,), transform=transform)

    train_data_loader = torch.utils.data.DataLoader(train, batch_size=16, num_workers=3, shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=16, num_workers=3, shuffle=True)

    # test = ImageFolder('/home/felix/pj/lung_classify/input/')

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 20):
        epoch_loss, epoch_accuracy = fit(epoch, model, train_data_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, model, valid_data_loader, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        ex.log_scalar('train_losses', epoch_loss)
        ex.log_scalar('train_accuracy', epoch_accuracy)
        ex.log_scalar('val_losses', val_epoch_loss)
        ex.log_scalar('val_accuracy', val_epoch_accuracy )




###### sacred begin
from sacred import Experiment
from easydict import EasyDict as edict
from sacred.observers import MongoObserver
from sacred import SETTINGS

ex = Experiment('lung')
db_url = 'mongodb://sample:password@10.10.20.103:27017/db?authSource=admin'
ex.observers.append(MongoObserver(url=db_url, db_name='db'))
SETTINGS.CAPTURE_MODE = 'sys'

@ex.config
def my_config():
    conf_name = None
    fold = -1


@ex.command()
def main(_config):

    config = edict(_config)
    conf_name = config.conf_name
    valid_fold = int(config.fold)

    train(valid_fold, conf_name)

if __name__ == '__main__':

    """"
    python -u customer/classify_ctm.py main with conf_name=5cls_resnet34  fold=0
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