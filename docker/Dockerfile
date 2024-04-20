FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

RUN apt-get update -y

RUN apt-get install -y python3 python3-pip 

RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libsm6 libgl1-mesa-glx

RUN pip install --upgrade pip

WORKDIR /pipeline/

ADD ../requirements.txt .
RUN pip3 install -r requirements.txt

COPY .. /pipeline/


