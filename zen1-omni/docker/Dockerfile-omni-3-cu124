# Dockerfile of qwenllm/qwen3-omni:3-cu124

ARG CUDA_VERSION=12.4.0
ARG from=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

FROM ${from} as base

ARG DEBIAN_FRONTEND=noninteractive
RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    vim \
    libsndfile1 \
    ccache \
    software-properties-common \
    ffmpeg \
&& rm -rf /var/lib/apt/lists/*
EOF

RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.1/cmake-3.26.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.1/bin/* /usr/local/bin

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git lfs install

FROM base as dev

WORKDIR /

RUN mkdir -p /data/shared/Qwen3-Omni

WORKDIR /data/shared/Qwen3-Omni/

FROM dev as bundle_req

FROM bundle_req as bundle_vllm

ARG BUNDLE_FLASH_ATTENTION=true

ENV MAX_JOBS=32
ENV NVCC_THREADS=2
ENV CCACHE_DIR=/root/.cache/ccache

ARG BUNDLE_VLLM=true

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_VLLM" = "true" ]; then \
    mkdir -p /data/shared/code \
        && cd /data/shared/code \
        && git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git \
        && cd vllm \
        && pip3 install -r requirements/build.txt \
        && pip3 install -r requirements/cuda.txt \
        && pip3 install . -v --no-build-isolation \
        && cd /data/shared/Qwen3-Omni \
        && rm -rf /data/shared/code/vllm; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip pip3 install git+https://github.com/huggingface/transformers

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
        pip install flash-attn --no-build-isolation; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip pip3 install networkx==3.4.2
RUN --mount=type=cache,target=/root/.cache/pip pip3 install accelerate==1.10.1 qwen-omni-utils huggingface_hub[cli] modelscope_studio modelscope

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
    gradio==5.44.1 \
    gradio_client==1.12.1 \
    soundfile==0.13.1 \
    librosa \
    av

RUN rm -rvf /root/.cache/pip

EXPOSE 80
