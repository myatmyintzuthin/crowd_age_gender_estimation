"""
@file      : yolox_m_person.py

@author    : Myat Myint Zu Thin
@date      : 2024/03/29
"""

import os
from models.yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = ""
        self.train_ann = ""
        self.val_ann = ""
        self.num_classes = 80

        self.max_epoch = 10
        self.data_num_workers = 1
        self.eval_interval = 1

        self.nmsthre = 0.45
        self.test_conf = 0.25
        self.test_size = (640, 640)
