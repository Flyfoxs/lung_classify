import logging
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

from fastai.callbacks import EarlyStoppingCallback,SaveModelCallback
import pydevd_pycharm

from easydict import EasyDict as edict
import yaml, os


from sacred import Experiment
from easydict import EasyDict as edict
from sacred.observers import MongoObserver



version = '13'

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




def train(valid_fold , conf_name):

    f = open(f'./configs/{conf_name}.yaml')
    conf = edict(yaml.load(f))

    class_cnt = conf.class_cnt
    backbone_name = conf.backbone
    unfreeze = conf.unfreeze if 'unfreeze' in conf else False
    epoch = 5

    assert int(valid_fold) <= 4
    # batch_id = str(round(time.time()))
    backbone = get_backbone(backbone_name)

    df = pd.read_csv('./input/train.csv', names=['file_name', 'label'])
    df['fold'] = df.file_name%5
    df['file_name'] = df.file_name.astype('str')+'.jpg'

    #print(df.head(), df.shape)
    if class_cnt <= 2:
        df.label = np.where(df.label>=1, 1, 0)

    data = (ImageList.from_df(df, './input/train/', )
             .split_by_idx(df.loc[df.fold == valid_fold].index)
             # split_by_valid_func(lambda o: int(os.path.basename(o).split('.')[0])%5==i)
             .label_from_df()
             # .add_test_folder('./input/test')
             .transform(get_transforms(), size=200)
             .databunch(bs=16)).normalize(imagenet_stats)

    test_data = ImageList.from_folder(path="./input/test")

    data.add_test(test_data)

    #data.show_batch(rows=3, figsize=(15,15))

    learn = cnn_learner(data, backbone, metrics=[ accuracy])

    checkpoint_name = f'{backbone()._get_name()}_f{valid_fold}'
    callbacks = [EarlyStoppingCallback(learn, monitor='accuracy', min_delta=1e-5, patience=5),
                 SaveModelCallback(learn, monitor='accuracy', name=checkpoint_name, every='improvement')
                 ]

    print(f'=====Fold:{valid_fold}, Total epoch:{epoch}, {conf_name}, backbone:{backbone_name}=========')

    if unfreeze:
        learn.freeze_to(-3)

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

    oof_file = f'./output/stacking/{version}_{host_name[:5]}_{conf_name}_f{valid_fold}_s{best_score:6.5f}_val{val_len}_trn{train_len}.h5'

    print(f'Stacking file save to:{oof_file}')
    save_stack_feature(oof_val, oof_test, oof_file)



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
    # pydevd_pycharm.settrace('192.168.1.101', port=1234, stdoutToServer=True, stderrToServer=True)
    config = edict(_config)
    conf_name = config.conf_name
    valid_fold = int(config.fold)

    train(valid_fold, conf_name)

if __name__ == '__main__':

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