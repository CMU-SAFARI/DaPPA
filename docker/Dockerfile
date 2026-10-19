# Build environment for DaPPA: Ubuntu 20.04 with the UPMEM SDK 2023.1.0.
#
#   docker build -t dappa -f docker/Dockerfile .
#   docker run --rm -it -v "$PWD":/dappa -w /dappa dappa

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libedit2 \
        libelf1 \
        libnuma1 \
        libpython3.8 \
        libxml2 \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# CMake >= 3.21 for CMakePresets.json (Ubuntu 20.04 ships 3.16).
RUN pip3 install --no-cache-dir "cmake>=3.21,<4"

# UPMEM no longer distributes this SDK release; use the copy archived by SAFARI.
RUN git clone --depth 1 https://github.com/CMU-SAFARI/UPMEM.git /opt/upmem && \
    rm -rf /opt/upmem/.git

ENV UPMEM_HOME=/opt/upmem/upmem-2023.1.0-Linux-x86_64
ENV PATH=${UPMEM_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${UPMEM_HOME}/lib
