FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y cmake build-essential libvulkan1

WORKDIR /neuralnet
COPY . .
RUN rm -rf build
RUN cmake --preset default
RUN cmake --build --preset default

ENTRYPOINT [ "/bin/bash", "/neuralnet/network" ]