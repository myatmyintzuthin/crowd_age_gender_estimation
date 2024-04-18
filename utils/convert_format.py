"""
@file      : convert_format.py

@author    : Myat Myint Zu Thin
@date      : 2024/04/03
"""
import torch

class Yolov8Format:
    def __init__(self, image, results) -> None:
        self.names = {0: "person", 1: "face"}
        self.orig_image = image
        self.boxes = self.convert_bboxes(results)
        self.probs = None

    def __len__(self):
        return len(self.boxes)

    def convert_bboxes(self, results):
        Bboxes = []
        for result in results:
            bbox = Bbox(
                result["person"],
                result["person_cls"],
                result["person_score"],
                result["orig_shape"],
            )
            Bboxes.append(bbox)
            if result["face"] != None:
                bbox = Bbox(
                result["face"],
                result["face_cls"],
                result["face_score"],
                result["orig_shape"],
                )
                Bboxes.append(bbox)
        return Bboxes

class Bbox:
    def __init__(self, bboxes, cls, score, orig_shape) -> None:

        xyxy = [float(i) for i in bboxes]
        bboxes.extend([round(score, 2), cls])
        data = [float(i) for i in bboxes]
        
        self.id = None
        self.boxes = torch.tensor([data], device='cuda')
        self.cls = torch.tensor([cls], device='cuda')
        self.conf = torch.tensor([score], device='cuda')
        self.data = torch.tensor([data], device='cuda')
        self.onlybox = torch.tensor([xyxy], device='cuda')
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """Return the boxes in xyxy format."""
        return self.onlybox

    
