ARG CUDA="9.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install ninja yacs cython matplotlib opencv-python tqdm websockets yacs pyyaml

# Install PyTorch 1.0 Nightly
ARG CUDA
RUN conda install pytorch-nightly cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya

# Install TorchVision master
RUN git clone https://github.com/pytorch/vision.git \
 && cd vision \
 && python setup.py install

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && python setup.py install --cuda_ext --cpp_ext

#RUN rm -r ./CyberCV/ -f
# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/hkaraoguz/maskrcnn-benchmark.git -b docker \
              && mkdir CyberCV \ 
              && mv maskrcnn-benchmark CyberCV/maskrcnn \
              && cd CyberCV/maskrcnn \
              && python setup.py build develop

ENV PYTHONPATH "${PYTHONPATH}:/CyberCV"

RUN mkdir -p /root/.torch/models
COPY _detectron_35858933_12_2017_baselines_e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC_output_train_coco_2014_train%3Acoco_2014_valminusminival_generalized_rcnn_model_final.pkl /root/.torch/models

#RUN rm -r ./websocket_manager/ -f

RUN git clone  https://gitlab-ci-token:8j5LoqR9seTfKoF2bqJD@gitlab.com/hkaraoguz/websocket_manager.git      



WORKDIR /
ENTRYPOINT [ "python", "./websocket_manager/websocket_manager/src/websocket_manager_maskrcnn.py" ]

