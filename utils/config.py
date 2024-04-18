"""
@file      : config.py

@author    : Myat Myint Zu Thin
@date      : 2024/04/03
"""

from omegaconf import OmegaConf

class Config:
    def __init__(self, conf_path: str) -> None:
        self.__conf = OmegaConf.load(conf_path)
        
    @property
    def conf(self):
        return self.__conf

    @property
    def face_model_conf(self):
        return self.__conf.face_detection
    
    @property
    def person_model_conf(self):
        return self.__conf.person_detection
    
    @property
    def mivolo_conf(self):
        return self.__conf.mivolo

    

