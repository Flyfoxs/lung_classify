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

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("lung_train",)
cfg.DATASETS.TEST = ("lung_val_4", "lung_val_2",)

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2

cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.TEST.EVAL_PERIOD = 100


BATCH_SIZE = 128
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE  # faster, and good enough for this toy dataset (default: 512)
cfg.SOLVER.MAX_ITER = 50000


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


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


class _Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_evaluator(cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        ret = super().build_hooks()
        return ret


def train_model(resume=True):

    print(f'BATCH_SIZE={BATCH_SIZE}, MAX_ITER={cfg.SOLVER.MAX_ITER}')

    trainer = _Trainer(cfg)

    trainer.resume_or_load(resume=resume)

    trainer.train()

    trainer.storage.histories()


def gen_oof():
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    print(cfg.MODEL.WEIGHTS)
    ###cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99   # set the testing threshold for this model
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4  # set threshold for retinanet
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.1
    # cfg.TEST.DETECTIONS_PER_IMAGE = 2
    # cfg.DATASETS.TEST = ("lung_val", )
    predictor = DefaultPredictor(cfg)


if __name__ == '__main__':
    train_model()