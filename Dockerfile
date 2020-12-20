#
# Wav2vec 2.0
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi
#

FROM python:3.8.6-slim-buster

LABEL maintainer Loreto Parisi loreto@musixmatch.com

WORKDIR /python

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential

# Install fairseq
RUN git clone https://github.com/pytorch/fairseq --depth=1 && cd fairseq && \
    git fetch origin ac11107ed41cb06a758af850373c239309d1c961 && \
    git checkout ac11107ed41cb06a758af850373c239309d1c961 && \
    pip install --editable .

# Install kenlm
RUN apt install -y --no-install-recommends \
    build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
RUN git clone https://github.com/kpu/kenlm --depth=1 && \
    cd kenlm && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=20 -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make -j 16

# Install Additional Dependencies (ATLAS, OpenBLAS, Accelerate, Intel MKL)
RUN apt-get install -y --no-install-recommends \
    libopenblas-dev libfftw3-dev

# Install wav2letter
RUN pip install packaging
RUN git clone https://github.com/facebookresearch/wav2letter -b v0.2 --depth=1 && \
    cd wav2letter/bindings/python && \
    # for CPU 0 for GPU 1
    USE_CUDA=0 \
    # will use Intel MKL for featurization but this may cause dynamic loading conflicts
    USE_MKL=0 \
    KENLM_ROOT_DIR=/python/kenlm/ \
    pip install -e .

# RUN pip install editdistance
RUN pip install soundfile && \
    apt install -y --no-install-recommends libsndfile1

COPY src/ .
CMD ["bash"]
