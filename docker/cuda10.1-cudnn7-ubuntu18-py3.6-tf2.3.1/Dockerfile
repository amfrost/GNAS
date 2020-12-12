FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

LABEL maintainer="amf85@cam.ac.uk"

WORKDIR ~/

RUN apt update && \
	apt install -y software-properties-common && \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt update && \
	apt install -y python3.6

RUN add-apt-repository universe && \
    apt install -y python3-pip && \
    pip3 install --upgrade pip

RUN pip3 install tensorflow-gpu

COPY gpu_test.py gpu_test.py

CMD ["bash"]
