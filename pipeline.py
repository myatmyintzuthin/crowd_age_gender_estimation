"""
@file      : pipeline.py

@author    : Myat Myint Zu Thin
@date      : 2024/04/02
"""
import os
import cv2
import argparse

from utils.config import Config
from utils.convert_format import Yolov8Format
from utils.plot import plot_results
from utils.logger import Logger
from predictors.mivolo_predictor import MivoloPredictor
from predictors.yolox_predictor import YOLOPredictor
from models.yolox.data.datasets.coco_classes import COCO_CLASSES
from models.mivolo.structures import PersonAndFaceResult

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="assets/test.jpg", type= str ,help="test image/video path")
    parser.add_argument("--conf", default="config/config.yaml", type= str, help="config file path")
    opt = parser.parse_args()
    return opt

class Pipeline:
    def __init__(self, input_path: str, conf_path: str) -> None:

        self.__results = []
        self.__conf = Config(conf_path)
        self.__logger = Logger().get_instance()
        self.__input_path = input_path

        if os.path.basename(input_path).split('.')[-1] in ["png", "jpg", "jpeg"]:
            self.__img_inf = True
        elif os.path.basename(input_path).split('.')[-1] in ["mp4"]:
            self.__img_inf = False
        else:
            self.__logger.error("Error: must be supported image or video format.")
            raise Exception("Invalid input")

        self.__save_path = self.__conf.conf.save_path
        self.__person_detector = self.__load_detector("person")
        self.__face_detector = self.__load_detector("face")
        self.__age_gender_model = self.__load_mivolo()
        self.__result_path = f'{self.__save_path}/{os.path.basename(self.__input_path)}'
        os.makedirs(self.__save_path, exist_ok=True)
        

    def run(self):

        if self.__img_inf:
            self.__org_image= cv2.imread(self.__input_path)
            result_image = self.__model_pipeline(self.__org_image)
            self.__save_image(result_image, self.__result_path)
        
        else:
            cap = cv2.VideoCapture(self.__input_path)

            if not cap.isOpened():
                self.__logger.error("Error: Couldn't open the video file.")
                return
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.__logger.info(f"Total frames: {total_frames}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.__result_path, fourcc, fps, (frame_width, frame_height))

            frame_cnt = 0
            while cap.isOpened():
                ret, frame = cap.read()

                if frame is None:
                    break

                if not ret:
                    break

                self.__logger.info(f"Processing: Frame {frame_cnt}")
                self.__org_image = frame
                result_image = self.__model_pipeline(frame)
                out.write(result_image)
                self.__results.clear()
                frame_cnt += 1

            cap.release()
            out.release()
            self.__logger.info(f"Visualized video saved in {self.__result_path}")


    def __save_image(self, image: cv2.Mat, save_path: str):

        cv2.imwrite(save_path, image)
        self.__logger.info(f"Visualized image saved in {save_path}")

    def __model_pipeline(self, image: cv2.Mat):
            
        self.__detect_person(image)
        self.__detect_face()
        processed_data = self.__convert_yolov8format()
        final_result = self.__age_gender_model.inference(image, PersonAndFaceResult(processed_data))
        viz_image = plot_results(image, final_result)
        return viz_image


    def __load_mivolo(self):
        conf = self.__conf.mivolo_conf
        predictor = MivoloPredictor(conf.model, conf.device, conf.fp16, conf.withperson, conf.disableface)
        return predictor

    def __load_detector(self, choice: str):
        if choice == "person":
            conf = self.__conf.person_model_conf
            predictor = YOLOPredictor(
                conf.model,
                conf.exp,
                COCO_CLASSES,
                conf.device,
                conf.fp16,
                save_result=False,
            )
        else:
            conf = self.__conf.face_model_conf
            predictor = YOLOPredictor(
                conf.model,
                conf.exp,
                conf.cls,
                conf.device,
                conf.fp16,
                save_result=False,
            )
        return predictor

    def __detect_person(self, image_path: cv2.Mat):
        results, img_info = self.__person_detector.inference(
            image_path, filter_cls=True
        )
        self.__postproc_person(results, img_info)

    def __detect_face(self):
        face_cnt = 0
        for idx, person_item in enumerate(self.__results):
            face_results, img_info = self.__face_detector.inference(
                person_item["person_img"]
            )
            face_cnt += self.__postproc_face(idx, face_results, img_info, person_item["person"])
        self.__logger.info(f"{face_cnt} faces detected.")

    def __postproc_person(self, results, img_info, conf=0.3):

        ratio = img_info["ratio"]
        image = img_info["raw_img"]
        results = results[0].cpu()
        results[:, 0:4] /= ratio

        results_list = results.tolist()
        for bbox in results_list:
            one_person = {}
            x0 = int(bbox[0])
            y0 = int(bbox[1])
            x1 = int(bbox[2])
            y1 = int(bbox[3])
            score = bbox[4] * bbox[5]
            
            if score < conf or x0 < 0 or y0 < 0:
                continue

            crop = image[y0:y1, x0:x1]
            one_person["person"] = [x0, y0, x1, y1]
            one_person["person_cls"] = 0
            one_person["person_score"] = bbox[4] * bbox[5]
            one_person["person_img"] = crop
            one_person["orig_shape"] = (self.__org_image.shape[0], self.__org_image.shape[1])
            self.__results.append(one_person)
        self.__logger.info(f"{len(self.__results)} persons detected.")

    def __postproc_face(self, idx, results, img_info, person_bbox, conf=0.5):

        if results[0] is None:
            self.__results[idx]['face'] = None
            return 0

        face_cnt = 0
        ratio = img_info["ratio"]
        results = results[0].cpu()
        results[:, 0:4] /= ratio
        results_list = results.tolist()

        for bbox in results_list:
            x0 = int(bbox[0])
            y0 = int(bbox[1])
            x1 = int(bbox[2])
            y1 = int(bbox[3])
            
            score = bbox[4] * bbox[5]
            
            if score < conf:
                self.__results[idx]['face'] = None
                continue

            x0, y0, x1, y1 = self.__resize_aspect(x0, y0, x1, y1, person_bbox)

            self.__results[idx]['face'] = [x0,y0,x1,y1]
            self.__results[idx]['face_cls'] = 1
            self.__results[idx]['face_score'] = bbox[4] * bbox[5]
            face_cnt += 1
        return face_cnt
            
    def __convert_yolov8format(self):
        format_converter = Yolov8Format(self.__org_image, self.__results)
        return format_converter

    def __resize_aspect(self, x0, y0, x1, y1, person_bbox):

        new_x_min = int(x0) + person_bbox[0]
        new_y_min = int(y0) + person_bbox[1]
        new_x_max = int(x1) + person_bbox[0]
        new_y_max = int(y1) + person_bbox[1]
        return new_x_min, new_y_min, new_x_max, new_y_max


if __name__ == "__main__":

    opt = get_args()
    pipeline = Pipeline(opt.input, opt.conf)
    pipeline.run()
