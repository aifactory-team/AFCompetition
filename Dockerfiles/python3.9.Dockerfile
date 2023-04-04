FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Ubuntu 20.04 required
RUN apt-get update --fix-missing && \
    apt-get install -y vim apt-utils && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get install -y gcc make

# Default Python3.9
RUN apt install -y python3.9-dev && \
    apt-get install -y python3.9-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    echo 'alias python3=python3.8' >> ~/.bashrc && \
    echo 'alias pip3=pip3.8' >> ~/.bashrc && \
    echo 'alias python=python3.9' >> ~/.bashrc && \
    echo 'alias pip=pip3.9' >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc" && \
    rm -rf get-pip.py

# Data science
RUN pip install tensorflow==2.11.1
RUN pip install keras-cv
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install albumentations
RUN pip install transformers
RUN pip install scikit-learn
RUN pip install numpy
RUN pip install Cython
RUN pip install pycocotools
RUN pip install pandas
RUN pip install jupyter
RUN pip install notebook
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install plotly
# Image processing
RUN pip install Pillow
RUN pip install opencv-python
RUN pip install scikit-image
# AIF submit required
RUN pip install -U aifactory
RUN pip install -U gdown
RUN pip install ipynbname==2021.3.2
RUN pip install requests
RUN pip install gradio
RUN pip install fastapi
RUN pip install "uvicorn[standard]"
RUN pip install python-multipart
RUN pip install altair
RUN pip install psutil==5.8.0
# # Extra
# RUN pip install langchain==0.0.123
# RUN pip install paper-qa==0.1.0
# RUN pip install openai
