from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation.coco_evaluation import COCOEvaluator

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from functools import lru_cache

import os
import numpy as np
import json
import cv2
import random



import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import os
from task_distribute.locker import task_locker
from file_cache.cache import file_cache, logger, timed

#from tqdm._tqdm_notebook import tqdm_notebook as  tqdm

from tqdm import tqdm, tqdm_notebook
from glob import glob
import itertools
import numpy as np
from easydict import EasyDict as edict

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

version = 'd03'

@lru_cache()
def get_anotation_dicts(fold_name=None, box_cnt=None):
    fold_map = {
        'train': (0, 1, 2, 3),
        'val_0': (0,),
        'val_1': (1,),
        'val_2': (2,),
        'val_3': (3,),
        'val_4': (4,),
    }
    fold = fold_map.get(fold_name)
    cls = pd.read_csv('./input/train.csv', names=['filename', 'label'])  # .sample(100)
    cls['fold'] = cls.filename % 5
    if fold:
        cls = cls.loc[cls.fold.isin(fold)]
    if box_cnt is not None:
        df = pd.read_csv('./input/train_bboxes.csv')
        tmp = df.groupby('filename').filename.count().sort_values()
        if box_cnt == 0:
            cls = cls.loc[~cls.filename.isin(tmp.index)]
        else:
            tmp = tmp.loc[tmp == box_cnt]
            cls = cls.loc[cls.filename.isin(tmp.index)]

    bbox = pd.read_csv('./input/train_bboxes.csv')

    dataset_dicts = []
    for idx, v in tqdm(cls.iterrows(), desc=f'gen_annotaion_{str(fold_name)}', total=len(cls)):
        record = {}
        filename = v["filename"]
        # filename = os.path.join(img_dir, v["filename"])
        height, width = 1024, 1024  # cv2.imread(f'input/train/{filename}.jpg').shape[:2]
        # print(height, width)

        record["file_name"] = f'input/train/{filename}.jpg'
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for _, anno in bbox.loc[bbox.filename == v["filename"]].iterrows():
            #             assert not anno["region_attributes"]
            #             anno = anno["shape_attributes"]
            #             px = anno["all_points_x"]
            #             py = anno["all_points_y"]
            #             poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            #             poly = [p for x in poly for p in x]

            if len(anno) > 0:
                # print(anno)
                obj = {
                    "bbox": [anno.x, anno.y, anno.width, anno.height],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": 0,
                    "thing_classes": 'bz',
                    "iscrowd": 0
                }
                objs.append(obj)
        if objs:
            record["annotations"] = objs

        if objs or box_cnt == 0:
            dataset_dicts.append(record)
    print(f'dataset_dicts={len(dataset_dicts)} with fold:{fold}, box_cnt:{box_cnt}')
    return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog
DatasetCatalog.clear()
for d in ["train", "val_0", "val_1", "val_2", "val_3", "val_4"]:
    name = "lung_" + d
    if name not in DatasetCatalog.list():
        print(d)
        DatasetCatalog.register(name, lambda d=d: get_anotation_dicts(d))
        MetadataCatalog.get(name).set(thing_classes=["bz"])
metadata = MetadataCatalog.get("lung_train")

def save_stack_feature(train: pd.DataFrame, test: pd.DataFrame, file_path):
    train.to_hdf(file_path, 'train', mode='a')
    test.to_hdf(file_path, 'test', mode='a')
    logger.info(f'OOF file save to :{file_path}')
    return train, test


class _Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_evaluator(cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        ret = super().build_hooks()
        return ret

    def train(self, patience=3):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        self.iter = start_iter = self.start_iter
        max_iter = self.max_iter = self.cfg.SOLVER.MAX_ITER

        from detectron2.utils.events import EventStorage
        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                print('start_iter, max_iter', start_iter, max_iter)
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    if (self.iter+1) % self.cfg.TEST.EVAL_PERIOD == 0 and self.early_stop(patience):
                        break
            finally:
                self.after_train()

    def early_stop(self, patience):

        logger = logging.getLogger(__name__)

        #logger.info(dir(self.storage))

        # if 'history' not in dir(self.storage) or 'historys' not in dir(self.storage):
        #     return False

        val_data_name = self.cfg.DATASETS.TEST[0]

        if len(self.cfg.DATASETS.TEST) == 1:
            val_name = 'bbox/AP50'
        else:
            val_name = f'{val_data_name}/bbox/AP50'
        #logger.info(f'==========={val_name}')
        his_dict = self.storage.histories()
        tmp = his_dict.get(val_name).values()
        logger.info(f'Score List@{self.iter}===:{tmp}')

        # Keep only score remove inter sn
        tmp = [item[0] for item in tmp]

        self.best_score = np.max(tmp)
        if tmp[-1]>=np.max(tmp) :
            self.checkpointer.save(self.cfg.best_model)

        if np.max(tmp[-patience:]) < np.max(tmp) :
            #最大值在不在最后几次评估
            logger.info(f'Early stop with===:{patience}, {tmp}')
            return True
        else:
            return False


def get_oof(cfg, data_type='val_4'):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"{cfg.best_model}.pth")
    print(cfg.MODEL.WEIGHTS)
    ###cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99   # set the testing threshold for this model
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4  # set threshold for retinanet
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.2
    # cfg.TEST.DETECTIONS_PER_IMAGE = 2


    if data_type == 'test':
        file_list = list(glob('./input/test/*.jpg'))
    else:
        # dataset_dicts = get_anotation_dicts(data_type)
        # file_list = [item["file_name"] for item in dataset_dicts]
        val_fold = int(data_type[-1])
        file_list = list(glob('./input/train/*.jpg'))
        file_list = [file for file in file_list if int(file.split('/')[-1].split('.')[0]) % 5 == val_fold]

    predictor = DefaultPredictor(cfg)
    res = []
    for file_name in tqdm(file_list, desc=data_type):
        # Predict Result
        file_name = file_name
        im = cv2.imread(file_name)
        outputs = predictor(im)
        res.append({'file_name': file_name, 'score': outputs.get('instances').get('scores').cpu().numpy()})

    df = pd.DataFrame(res)
    return df

def get_local_cfg(val_fold = 4, BATCH_SIZE = 128):


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("lung_train",)

    val_name = f"lung_val_{val_fold}"
    cfg.best_model = f'best_{val_name}'
    cfg.DATASETS.TEST = (val_name,)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.TEST.EVAL_PERIOD = 200

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE  # faster, and good enough for this toy dataset (default: 512)
    cfg.SOLVER.MAX_ITER = 50000

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


###### sacred begin
from sacred import Experiment
from easydict import EasyDict as edict
from sacred.observers import MongoObserver
from sacred import SETTINGS

ex = Experiment('lung_de')
db_url = 'mongodb://sample:password@10.10.20.103:27017/db?authSource=admin'
ex.observers.append(MongoObserver(url=db_url, db_name='db'))
SETTINGS.CAPTURE_MODE = 'sys'

@ex.config
def my_config():
    conf_name = None
    fold = -1


@ex.command()
def main(_config):

    val_fold = edict(_config).fold
    batch_size = 128
    resume = True

    cfg = get_local_cfg(val_fold, batch_size)

    print(f'BATCH_SIZE={batch_size}, MAX_ITER={cfg.SOLVER.MAX_ITER}')

    trainer = _Trainer(cfg)

    trainer.resume_or_load(resume=resume)

    trainer.train()


    # Begin to gen OOF file
    oof_val = get_oof(cfg, f'val_{val_fold}')
    oof_test = get_oof(cfg, 'test')

    os.makedirs('./output/stacking/', exist_ok=True)
    import socket
    host_name = socket.gethostname()
    conf_name = 'retinanet'
    best_score = trainer.best_score
    oof_file = f'./output/stacking/{version}_{host_name[:5]}_{conf_name}_f{val_fold}_s{best_score:6.5f}_val{len(oof_val)}.h5'

    print(f'Stacking file save to:{oof_file}')
    save_stack_feature(oof_val, oof_test, oof_file)


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
