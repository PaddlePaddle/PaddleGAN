# Based on a older version of https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libboost-all-dev \
         python-qt4 \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda config --set always_yes yes --set changeps1 no && conda update -q conda 
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install face-alignment package
WORKDIR /workspace
RUN chmod -R a+w /workspace
RUN git clone https://github.com/1adrianb/face-alignment
WORKDIR /workspace/face-alignment
RUN pip install -r requirements.txt
RUN python setup.py install
