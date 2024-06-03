FROM ubuntu:latest

LABEL org.opencontainers.image.source=https://github.com/heyallnorahere/neuralnet
LABEL org.opencontainers.image.description="neuralnet linux build"
LABEL org.opencontainers.image.licenses=Apache-2.0

RUN apt-get update
RUN apt-get install -y git cmake build-essential libvulkan1

WORKDIR /neuralnet
COPY . .
RUN rm -rf build data
RUN cmake --preset docker
RUN cmake --build --preset docker

ENTRYPOINT [ "/bin/bash", "/neuralnet/network" ]