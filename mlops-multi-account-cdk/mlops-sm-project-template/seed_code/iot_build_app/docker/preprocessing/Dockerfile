FROM public.ecr.aws/lts/ubuntu:22.04_stable

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get -y install --no-install-recommends \
    python3-pip \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


RUN pip3 install \
    albumentations \
    opencv-python-headless \
    tqdm \
    jsonlines

RUN mkdir -pv \
    /opt/ml/processing/input \
    /opt/ml/processing/input/train \
    /opt/ml/processing/input/test \
    /opt/ml/processing/input/val \
    /opt/ml/processing/source \
    /opt/ml/processing/model \
    /opt/ml/processing/output \
    /opt/ml/processing/output/train \
    /opt/ml/processing/output/val \
    /opt/ml/processing/output/test
    
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
