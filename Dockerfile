FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /workspace

RUN apt-get update
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
