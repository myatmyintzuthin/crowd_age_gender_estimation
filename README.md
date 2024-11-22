# Age and Gender Estimation in Crowded Scenes

## Overview
Most of the age and gender estimation models works on one or two person image. In this repo, age and gender estimation pipeline which is able to estimate in crowded scenes is built.

Currently, Mivolo (SOTA) model in age and gender estimation is trained on Lagenda Dataset. The model itself is very good at estimating if we can give person and face detected bounding boxes. The original detector used in Mivolo, YOLOv8 is not able to detect person and faces on crowded scenes.
So, the detector part is modified.

New pipeline structure:

<img src="assets/pipeline.png" width='460' height='300'/> 

## Host System Specification
* Ubuntu - 22.04
* Cuda - 11.7
* Python - 3.10.12

## Environment Setup

Clone the repo:
```
$ git clone https://github.com/myatmyintzuthin/crowd_age_gender_estimation.git
$ cd crowd_age_gender_estimation
```
### Install with requirements.txt
Create python virtual environment:
```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Install with Docker
build docker
```
$ make docker-build
```
run docker
```
$ make docker-run
```

### Run pipeline
```
$ python pipeline.py --input {image path/video path} --conf config/config.yaml
```

## Model 

| model | type  | weight |
| :---: | :---: | :----: |
| Person Detector | YOLOX-m (pretrained) | [yolox_m.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) | 
| Face Detector | YOLOX-s (custom) | [yolox_s_face.pth](https://github.com/myatmyintzuthin/crowd_age_gender_estimation/releases/tag/face_detection_weight) | 
| Age&Gender Estimator | Mivolo_d1 (pretrained) | [mivolo_imdb_cross_person_4.22_99.46.pth.tar](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view?usp=drive_link) | 

Detail face detector training : [docs/faceDetectorTraining.md](docs/faceDetectorTraining.md)


## Sample Output
| Mivolo | Crowd Pipeline |   
| :---:  | :---: | 
| <img src="assets/japan1_mivolo.jpg"/> | <img src="assets/japan1_pipeline.png"/>   |
| <img src="assets/japan4_mivolo.jpg"/> | <img src="assets/japan4_pipeline.png"/>      |

## Video Output
<img src="assets/tokyo_walk.gif"/>

# References

```
@article{mivolo2023,
   Author = {Maksim Kuprashevich and Irina Tolstykh},
   Title = {MiVOLO: Multi-input Transformer for Age and Gender Estimation},
   Year = {2023},
   Eprint = {arXiv:2307.04616},
}
```
```
@article{mivolo2024,
   Author = {Maksim Kuprashevich and Grigorii Alekseenko and Irina Tolstykh},
   Title = {Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation},
   Year = {2024},
   Eprint = {arXiv:2403.02302},
}
```
```
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```





