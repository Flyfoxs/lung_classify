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

from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback, TrackerCallback
from efficientnet_pytorch import EfficientNet

from easydict import EasyDict as edict
import yaml, os


from sacred import Experiment
from easydict import EasyDict as edict
from sacred.observers import MongoObserver



version = 'n16'

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


def train(valid_fold , conf_name):


    file_name = conf_name
    if not os.path.exists(file_name):
        print(f'Can not find file:{file_name}')
        file_name = f'./configs/{file_name}.yaml'
    f = open(file_name)
    conf = edict(yaml.load(f))


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

    #learn = cnn_learner(data, backbone, metrics=[ accuracy])
    from customer.create_network import get_network
    model = get_network(backbone_name, data.c)
    learn = Learner(data, model, metrics=[ accuracy])

    ch_prefix = os.path.basename(file_name)
    checkpoint_name = f'{ch_prefix}_f{valid_fold}'
    callbacks = [EarlyStoppingCallback(learn, monitor='accuracy', min_delta=1e-5, patience=5),
                 SaveModelCallback(learn, monitor='accuracy', name=checkpoint_name, every='improvement'),
                 Recorder_scared(ex, learn )
                 ]

    print(f'=====Fold:{valid_fold}, Total epoch:{epoch}, {conf_name}, backbone:{backbone_name}=========')

    if unfreeze:
        learn.freeze_to(-2)

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
    python -u customer/classify.py main with conf_name=5cls_resnet34  fold=0
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