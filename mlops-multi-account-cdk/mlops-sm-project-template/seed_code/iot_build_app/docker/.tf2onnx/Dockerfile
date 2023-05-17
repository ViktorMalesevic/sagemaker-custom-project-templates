# SageMaker TF2.11 image
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.11-cpu
FROM public.ecr.aws/lts/ubuntu:22.04_stable

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG TF_VERSION=2.11

RUN apt-get update \
 && apt-get install python3-pip -y \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ENV variable to be passed to SageMaker stage
ENV PIP=${PIP}
ENV PYTHON=${PYTHON}

# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8
# Python won’t try to write .pyc or .pyo files on the import of source modules
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH='/usr/local/lib:$LD_LIBRARY_PATH'

# update pip
RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# install TF2 and ONNX
RUN ${PIP} install --no-cache-dir \
    "protobuf>=3.19.5,<3.20" \
    tensorflow-cpu==$TF_VERSION \
    tf2onnx 

# prepare folders
RUN mkdir -pv \
    /opt/ml/processing/input \
    /opt/ml/processing/output \
    /opt/ml/processing/model \
    /opt/ml/processing/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY onnx_converter.py /opt/ml/code/onnx_converter.py

# Defines onnx_converter.py as script entrypoint 
# ENV PROGRAM=onnx_converter.py

ENTRYPOINT ["python3", "/opt/ml/code/onnx_converter.py"]
