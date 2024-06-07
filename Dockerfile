FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

LABEL org.opencontainers.image.source=https://github.com/heyallnorahere/neuralnet
LABEL org.opencontainers.image.description="neuralnet linux build"
LABEL org.opencontainers.image.licenses=Apache-2.0

RUN apt-get update
RUN apt-get install -y git cmake build-essential libvulkan1 mesa-utils

WORKDIR /neuralnet
COPY vendor vendor
COPY CMakePresets.json CMakeLists.txt network ./
RUN cmake --preset docker -DNN_SKIP_NEURALNET=ON
RUN cmake --build --preset docker-deps

COPY src src
RUN cmake --preset docker -DNN_SKIP_NEURALNET=OFF
RUN cmake --build --preset docker-build

ENTRYPOINT [ "/bin/bash", "/neuralnet/network" ]