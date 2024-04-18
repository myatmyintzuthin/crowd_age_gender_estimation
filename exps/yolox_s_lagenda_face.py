"""
@file      : yolox_s_lagenda_face.py

@author    : Myat Myint Zu Thin
@date      : 2024/04/02
"""

import os
from models.yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/lagenda_face_dataset"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.num_classes = 1
        self.multiscale_range = 0
        self.save_history_ckpt = False

        self.max_epoch = 20
        self.data_num_workers = 4
        self.eval_interval = 1

        self.nmsthre = 0.45
        self.test_conf = 0.25
        self.test_size = (640, 640)
