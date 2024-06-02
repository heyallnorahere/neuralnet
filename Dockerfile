FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL org.opencontainers.image.source=https://github.com/heyallnorahere/neuralnet
LABEL org.opencontainers.image.description="neuralnet linux build"
LABEL org.opencontainers.image.licenses=Apache-2.0

RUN apt-get update
RUN apt-get install -y cmake build-essential libvulkan1

WORKDIR /neuralnet
COPY . .
RUN rm -rf build
RUN cmake --preset default
RUN cmake --build --preset default

ENTRYPOINT [ "/bin/bash", "/neuralnet/network" ]