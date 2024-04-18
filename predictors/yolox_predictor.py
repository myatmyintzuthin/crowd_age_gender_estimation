'''
@file      : yolox_predictor.py

@author    : Myat Myint Zu Thin
@date      : 2024/04/01
'''

import os
import cv2
import torch

from models.yolox.data.data_augment import ValTransform
from models.yolox.data.datasets.coco_classes import COCO_CLASSES
from models.yolox.exp import get_exp
from models.yolox.utils import postprocess, vis
from utils.logger import Logger

class YOLOPredictor(object):

    def __init__(self, model_path: str, exp_path: str, cls_name: list, device='cpu', fp16=False, save_result=False) -> None:

        self.__fp16 = fp16
        self.__device = device
        self.__cls_name = cls_name
        self.__save_result = save_result
        self.__preproc = ValTransform(legacy=False)
        self.__logger = Logger().get_instance()

        self.__init_exp(exp_path)
        self.__model = self.__init_model(model_path) 
        
    @property
    def confthre(self):
        return self.__confthre
    
    @property
    def vis_folder(self):
        return self.__vis_folder

    def __init_exp(self, exp_path: str):
        
        self.__exp = get_exp(exp_path)
        filename = os.path.join(self.__exp.output_dir, self.__exp.exp_name)
        os.makedirs(filename, exist_ok=True) if self.__save_result else None

        if self.__save_result: 
            self.__vis_folder = os.path.join(filename, "vis_res")
            os.makedirs(self.__vis_folder, exist_ok=True)
        
        self.__confthre = self.__exp.test_conf
        self.__nmsthre = self.__exp.nmsthre
        self.__test_size = self.__exp.test_size
        self.__num_classes = self.__exp.num_classes
        
    def __init_model(self, model_path: str):

        if model_path is None:
            self.__logger.error("[Error] model weight does not exits!")
            return 
        else:
            yolox_model = self.__exp.get_model()
            if self.__device == "gpu":
                yolox_model.cuda()
                if self.__fp16:
                    yolox_model.half()  # to FP16
            yolox_model.eval()
            check_point = torch.load(model_path, map_location='cpu')
            yolox_model.load_state_dict(check_point['model'])
            self.__logger.info(f"Loaded '{os.path.basename(model_path)}' checkpoint")
            return yolox_model

    def inference(self, img_path: str, filter_cls = False):
        img_info = {"id": 0}
        if isinstance(img_path, str):
            img_info["file_name"] = os.path.basename(img_path)
            img = cv2.imread(img_path)
        else:
            img = img_path
            img_info["file_name"] = None

        height, width = img.shape[:2]
        
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.__test_size[0] / img.shape[0], self.__test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, __ = self.__preproc(img, None, self.__test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.__device == "gpu":
            img = img.cuda()
            if self.__fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.__model(img)
            outputs = postprocess(
                outputs, self.__num_classes, self.__confthre,
                self.__nmsthre, class_agnostic=True
            )
            if filter_cls:
                outputs = self.__filter_person(outputs)

        if self.__save_result:
            vis_image = self.visual(outputs[0], img_info, self.__confthre)
            save_file_name = os.path.join(self.__vis_folder, os.path.basename(img_path))
            cv2.imwrite(save_file_name, vis_image)
            self.__logger.info(f"Visualize result saved in {self.__vis_folder}")
        
        return outputs, img_info
    
    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.__cls_name)
        return vis_res
    
    def __filter_person(self, outputs):
        
        det_outputs = outputs[0]
        det_output = det_outputs[det_outputs[:, -1] == 0]
        outputs[0] = det_output
        return outputs

if __name__ == "__main__":

    image_path = 'assets\\test_person.jpg'
    face_model = 'weights\\yolox_m.pth'
    exp_path = 'Exps\\yolox_m_person.py'
    save_result = True

    predictor = YOLOPredictor(face_model, exp_path, COCO_CLASSES, device=0, fp16=True, save_result=True)

    results = predictor.inference(image_path, filter_cls = True)
    










