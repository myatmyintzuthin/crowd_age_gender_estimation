"""
@file      : plot.py

@author    : Myat Myint Zu Thin
@date      : 2024/04/04
"""
import cv2
import torch
from utils.color import RGBs
from utils.logger import Logger
from models.mivolo.structures import PersonAndFaceResult
logger = Logger().get_instance()

def plot_results(image: cv2.Mat, save_path: str, detected_bboxes: PersonAndFaceResult, image_inf: bool):

        colors_by_ind = {}
        results = detected_bboxes.yolo_results
        for face_ind, person_ind in detected_bboxes.face_to_person_map.items():
            if person_ind is not None:
                colors_by_ind[face_ind] = face_ind + 1
                colors_by_ind[person_ind] = face_ind + 1
            else:
                colors_by_ind[face_ind] = 0
        for person_ind in detected_bboxes.unassigned_persons_inds:
            colors_by_ind[person_ind] = 1
        
        conf=False
        gender_scores = False
        pred_boxes = results.boxes

        vis_image = image.copy()

        for bb_ind, (d, age, gender, gender_score) in enumerate(zip(pred_boxes, detected_bboxes.ages, detected_bboxes.genders, detected_bboxes.gender_scores)):
            c, conf, guid = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            name = ("" if guid is None else f"id:{guid} ") + results.names[c]
            label = (f"{name} {conf:.2f}" if conf else name) 
            if age is not None:
                label += f" {age:.1f}"
            if gender is not None:
                label += f" {'F' if gender == 'female' else 'M'}"
            if gender_scores and gender_score is not None:
                label += f" ({gender_score:.1f})"
            vis_image = draw_bbox(vis_image, d.xyxy.squeeze(), label, colors_by_ind[bb_ind])

        if image_inf:
            save_image(vis_image, save_path)
        

def draw_bbox(image, bbox, label, color_index):

        if isinstance(bbox, torch.Tensor):
            bbox = bbox.tolist()
        
        if 'face' in label:
            color = RGBs[17]          # Face color
        elif color_index == 1:
            color = RGBs[3]          # Person without face color
        else:   
            color = RGBs[0]           # Normal Person color

        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        background_width = text_size[0] + 5 
        background_height = text_size[1] + 5 

        image = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
        text_x = int(x0 + 5)
        text_y = int(y0 - 5) 
        if not 'face' in label:
            image = cv2.rectangle(image, (int(x0), int(y0)), (int(x0 + background_width), int(y0 - background_height)), color, -1)
            image = cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 

        return image

def save_image(image: cv2.Mat, save_path: str):

    cv2.imwrite(save_path, image)
    logger.info(f"Visualized image saved in {save_path}")