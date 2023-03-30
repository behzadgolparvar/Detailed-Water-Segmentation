# TrainingPointRend
# # **Install Dependencies**

# %%
# # install dependencies: 
# !pip install pyyaml==5.1
# # clone the repo in order to access pre-defined configs in PointRend project
# !git clone --branch v0.6 https://github.com/facebookresearch/detectron2.git detectron2_repo
# # install detectron2 from source
# !pip install -e detectron2_repo
# # See https://detectron2.readthedocs.io/tutorials/install.html for other installation options

# %% [markdown]
# # **Setup**

# %%
# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# %%
print(torch.version.cuda)

# %%
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend

import random, os

# %% [markdown]
# # **Import Datasets**

# %%
TrainJsonFilePath = os.path.join(os.getcwd(), 'Labels/Sea_Beach_others_Train.json')
ValidJsonFilePath = os.path.join(os.getcwd(), 'Labels/Sea_Beach_others_Valid.json')
dataset_folderPath = os.path.join(os.getcwd(), 'Labels/Dataset_COCOstuff_Beach')

# %%
from detectron2.data.datasets import register_coco_instances

register_coco_instances("BeachCOCO_Train", {}, TrainJsonFilePath, dataset_folderPath)

Train_metadata = MetadataCatalog.get("BeachCOCO_Train")
TrainD_dicts = DatasetCatalog.get("BeachCOCO_Train")

# %%
register_coco_instances("BeachCOCO_Valid", {}, ValidJsonFilePath, dataset_folderPath)

Valid_metadata = MetadataCatalog.get("BeachCOCO_Valid")
Valid_dicts = DatasetCatalog.get("BeachCOCO_Valid")

# %% [markdown]
# # **Training**

# %%
from detectron2.projects import point_rend
from detectron2.engine import DefaultTrainer

# %%
cfg = get_cfg()

point_rend.add_pointrend_config(cfg)
cfg.merge_from_file("../projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# %%
cfg.DATASETS.TRAIN = ("BeachCOCO_Train",)
cfg.DATASETS.TEST = ("BeachCOCO_Valid",)
cfg.DATALOADER.NUM_WORKERS = 2 # 4 is defualt but 2 was used in instructions   

cfg.SOLVER.GAMMA = 0.5
# The iteration number to decrease learning rate by GAMMA.
cfg.SOLVER.STEPS = (2000,)

# Save a checkpoint after every this number of iterations
cfg.SOLVER.CHECKPOINT_PERIOD = 2000

cfg.OUTPUT_DIR = './OutputModels'

cfg.INPUT.MASK_FORMAT = "bitmask"     #https://detectron2.readthedocs.io/modules/config.html
cfg.SOLVER.IMS_PER_BATCH = 2 #  is default but 2 was used in instructions
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 8001    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(Train_metadata.thing_classes)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

# %%
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models

cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_Beach_segment.pth")  # path to the model we just trained

# %%
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# %%
cfg['MODEL']['POINT_HEAD']['NUM_CLASSES'] = cfg['MODEL']['ROI_HEADS']['NUM_CLASSES']

# %%
import trainerPointRend

cfg.MODEL.DEVICE = 'cpu'
trainer = trainerPointRend.Trainer(cfg)

# %%

# trainer.resume_or_load(resume=False)
trainer.resume_or_load(resume=True)

# %%
trainer.train()

# %%
from detectron2.checkpoint import DetectionCheckpointer

checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
checkpointer.save("model_Beach_segment")  # 

# https://github.com/facebookresearch/detectron2/issues/958

# %%


# %%


# %%



