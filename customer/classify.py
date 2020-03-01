import logging

from customer.create_network import get_backbone

logging.basicConfig(level=logging.INFO)

import pandas as pd
import os
from task_distribute.locker import task_locker
from file_cache.cache import file_cache, logger, timed


from tqdm import tqdm, tqdm_notebook
from glob import glob
import itertools
import numpy as np
from easydict import EasyDict as edict

import warnings
warnings.filterwarnings("ignore")
import time
import pandas as pd



from fastai import *
from fastai.vision import *
from fastai.widgets import *

from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback, TrackerCallback
from efficientnet_pytorch import EfficientNet

from easydict import EasyDict as edict
import yaml, os


from sacred import Experiment
from easydict import EasyDict as edict
from sacred.observers import MongoObserver

from customer.create_network import get_network
import torchvision.models as to_models


version = 'sp01'

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




class Recorder_scared(Recorder):
    "A `LearnerCallback` that records epoch, loss, opt and metric data during training."
    _order=-10
    def __init__(self, ex, learn:Learner, add_time:bool=True, silent:bool=True):
        super().__init__(learn, add_time=add_time, silent=silent)
        self.ex = ex

    def on_epoch_end(self, epoch:int, num_batch:int, smooth_loss:Tensor,
                     last_metrics:MetricsList, **kwargs:Any)->bool:
        "Save epoch info: num_batch, smooth_loss, metrics."
        self.nb_batches.append(num_batch)
        if last_metrics is not None: self.val_losses.append(last_metrics[0])
        else: last_metrics = [] if self.no_val else [None]
        if len(last_metrics) > 1: self.metrics.append(last_metrics[1:])
        self.format_stats([epoch, smooth_loss] + last_metrics, epoch)

    def format_stats(self, stats:TensorOrNumList, epoch)->None:
        "Format stats before printing."
        str_stats = []
        for name,stat in zip(self.names,stats):
            if name != 'epoch':
                self.ex.log_scalar(name, round(float(stat), 4), step=epoch)


def get_eff_learner(data, backbone_name = 'efficientnet-b0'):
    eff = lambda dummy: EfficientNet.from_pretrained(backbone_name, num_classes=5)
    model = eff(True)
    model.requires_grad = False
    model._fc.requires_grad = True
    learn = Learner(data, model, metrics=[accuracy])
    return learn


def get_fastai_learn(data,  backbone_name = "resnet34" ):
    backbone_fn = getattr(to_models, backbone_name)
    learn = cnn_learner(data, backbone_fn,  metrics=[accuracy])
    return learn


def get_ens_learn(data,  backbone_name):
    net = get_network(backbone_name, data.c)
    learn = Learner(data, net, metrics=[accuracy])  #cnn_learner(data, backbone_fn, metrics=[accuracy])
    return learn

def get_test_learn(data,  backbone_name = "resnet34" ):
    #backbone_fn = getattr(to_models, backbone_name)
    from torchvision import models
    tmp = models.resnet34(True)
    tmp.fc.out_features = 5
    learn = Learner(data, tmp, metrics=[accuracy])
    return learn

def get_file_conf(file_name):
    if file_name is None:
        return edict()
    if not os.path.exists(file_name):
        #print(f'Can not find file:{file_name}')
        file_name = f'./configs/{file_name}.yaml'
    print(f'Find file conf:{file_name}')
    f = open(file_name)
    conf = edict(yaml.load(f))
    return conf


def train(sacred_conf):
    valid_fold = sacred_conf.fold
    image_size = sacred_conf.image_size
    conf = sacred_conf


    # class_cnt = 5
    backbone_name = conf.backbone
    unfreeze = True #conf.unfreeze if 'unfreeze' in conf else False
    epoch = 50

    assert int(valid_fold) <= 4
    # batch_id = str(round(time.time()))

    df = pd.read_csv('./input/train.csv', names=['file_name', 'label'])
    df['fold'] = df.file_name%5
    df['file_name'] = df.file_name.astype('str')+'.jpg'

    # #print(df.head(), df.shape)
    # if class_cnt <= 2:
    #     df.label = np.where(df.label>=1, 1, 0)

    # def get_transforms(do_flip: bool = True, flip_vert: bool = False, max_rotate: float = 10., max_zoom: float = 1.1,
    #                    max_lighting: float = 0.2, max_warp: float = 0.2, p_affine: float = 0.75,
    #                    p_lighting: float = 0.75, xtra_tfms: Optional[Collection[Transform]] = None) -> Collection[Transform]:
    #     "Utility func to easily create a list of flip, rotate, `zoom`, warp, lighting transforms."
    #     res = [rand_crop()]
    #     if do_flip:    res.append(dihedral_affine() if flip_vert else flip_lr(p=0.5))
    #     if max_warp:   res.append(symmetric_warp(magnitude=(-max_warp, max_warp), p=p_affine))
    #     if max_rotate: res.append(rotate(degrees=(-max_rotate, max_rotate), p=p_affine))
    #     if max_zoom > 1: res.append(rand_zoom(scale=(1., max_zoom), p=p_affine))
    #     if max_lighting:
    #         res.append(brightness(change=(0.5 * (1 - max_lighting), 0.5 * (1 + max_lighting)), p=p_lighting))
    #         res.append(contrast(scale=(1 - max_lighting, 1 / (1 - max_lighting)), p=p_lighting))
    #     #       train                   , valid
    #     return (res + listify(xtra_tfms)+zoom_crop(scale=0.1), zoom_crop(scale=0.1))

    data = (ImageList.from_df(df, './input/train/', )
             .split_by_idx(df.loc[df.fold == valid_fold].index)
             #.split_from_df('label')
             # split_by_valid_func(lambda o: int(os.path.basename(o).split('.')[0])%5==i)
             .label_from_df()
             # .add_test_folder('./input/test')
             .transform(get_transforms(), size=image_size)
             .databunch(bs=16)).normalize(imagenet_stats)
    test_data = ImageList.from_folder(path="./input/test")
    data.add_test(test_data)

    # backbone = get_backbone(backbone_name)

    #print(to_models.resnet34())
    # model_fun = to_models.resnet34
    # model_name = model_fun.__name__

    if ',' in backbone_name or isinstance(backbone_name, list):
        learn = get_ens_learn(data,backbone_name)
    elif 'eff' in backbone_name:
        learn = get_eff_learner(data, backbone_name)
    else:
        learn = get_fastai_learn(data, backbone_name)

    # learn = get_test_learn(data, backbone_name)


    model_name = backbone_name
    print(model_name, learn.model)

    # ch_prefix = os.path.basename(file_name)
    # checkpoint_name = f'{model_name}_f{valid_fold}'
    callbacks = [EarlyStoppingCallback(learn, monitor='accuracy', min_delta=1e-5, patience=5),
                 #SaveModelCallback(learn, monitor='accuracy', name=checkpoint_name, every='improvement'),
                 Recorder_scared(ex, learn )
                 ]

    print(f'=====Fold:{valid_fold}, Total epoch:{epoch}, {conf_name}, model_fun:{model_name}, image:{image_size} =========')

    learn.fit_one_cycle(epoch, callbacks=callbacks)

    oof_val = get_oof_df(learn, DatasetType.Valid)

    oof_test = get_oof_df(learn, DatasetType.Test)

    os.makedirs('./output/stacking/', exist_ok=True)
    import socket
    host_name = socket.gethostname()
    # score_list = np.array(learn.recorder.metrics)
    # best_epoch = np.argmax(score_list)
    # best_score = np.max(score_list)
    val_len = len(learn.data.valid_ds.items)
    train_len = len(learn.data.train_ds.items)

    from sklearn.metrics import accuracy_score
    best_score = accuracy_score(oof_val.iloc[:, :-1].idxmax(axis=1), oof_val.iloc[:, -1])

    conf_name_base = os.path.basename(conf_name)
    oof_file = f'./output/stacking/{version}_{host_name[:5]}_s{best_score:6.5f}_{conf_name_base}_f{valid_fold}_val{val_len}_trn{train_len}.h5'

    print(f'Stacking file save to:{oof_file}')
    save_stack_feature(oof_val, oof_test, oof_file)

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
    image_size = 200


@ex.command()
def main(_config):
    config = edict(_config)
    print('=====', config)
    train(config)

if __name__ == '__main__':

    """"
    python -u customer/classify.py main with conf_name=5cls_resnet34  fold=0
    python -u customer/classify.py main with conf_name=5cls_efficientnet-b0 fold=0 version=r1
    python -u customer/classify.py main with conf_name=ens_res_den_vgg fold=0 version=r1
    python -u customer/classify.py main with conf_name=5cls_efficientnet-b6 image_size=400 fold=0 version=r1

    python -u customer/classify.py main with backbone=efficientnet-b3 image_size=300 fold=0 version=r1
    python 
    """

    from sacred.arg_parser import get_config_updates
    import sys
    argv_conf, _ = get_config_updates(sys.argv[1:])

    conf_name = argv_conf.get('conf_name')
    real_conf = get_file_conf(conf_name)
    real_conf.update(argv_conf)
    real_conf.conf_name = conf_name or real_conf.backbone

    print(real_conf)
    if 'version' in real_conf : version = real_conf.get('version')

    locker = task_locker('mongodb://sample:password@10.10.20.103:27017/db?authSource=admin', remove_failed =9 , version=version)
    task_id = f'lung_{real_conf}'
    #pydevd_pycharm.settrace('192.168.1.101', port=1234, stdoutToServer=True, stderrToServer=True)
    with locker.lock_block(task_id=task_id) as lock_id:
        if lock_id is not None:
            ex.add_config({
                **real_conf,
                'lock_id': lock_id,
                'lock_name': task_id,
                'version': version,
                'GPU': os.environ.get('CUDA_VISIBLE_DEVICES'),
            })


            res = ex.run_commandline()