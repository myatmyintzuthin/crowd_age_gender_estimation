"""
@file      : mivolo_predictor.py

@author    : Myat Myint Zu Thin
@date      : 2024/04/02

"""

import cv2
import torch
from models.mivolo.model.mi_volo import MiVOLO

class MivoloPredictor(object):
    def __init__(self, model_path: str, device: int =0, fp16: bool=True, with_person: bool=False, disable_faces: bool = False ) -> None:
        
        self.__device = device
        self.__half = fp16 
        self.__with_person = with_person
        self.__disable_faces = disable_faces
        self.__verbose = False
        self.__model = self.__load_model(model_path)

    def __load_model(self, model_path: str):
        return MiVOLO(
            model_path,
            self.__device,
            self.__half,
            use_persons=self.__with_person,
            disable_faces=self.__disable_faces,
            verbose=self.__verbose,
        )

    def inference(self, image, detected_objects ):
        detected_objects = self.__model.predict(image, detected_objects)
        return detected_objects
        
if __name__ == '__main__':

    model_path = "weights\\mivolo_imdb_cross_person_4.22_99.46.pth.tar"
    predictor = MivoloPredictor(model_path, 0, fp16=True, with_person=True, disable_faces=False)
    
    

