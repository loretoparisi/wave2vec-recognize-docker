#
# Wav2vec 2.0
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi
#

FROM python:3.7.4-slim-buster

LABEL maintainer Loreto Parisi loreto@musixmatch.com

WORKDIR app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential


# Install python libraries
RUN pip3 install soundfile torchaudio sentencepiece

# Update apt-get & Install soundfile
RUN apt-get -y install apt-utils gcc libpq-dev libsndfile-dev

# for CPU 0 for GPU 1
ENV USE_CUDA=0
# will use Intel MKL for featurization but this may cause dynamic loading conflicts
ENV USE_MKL=0
ENV KENLM_ROOT_DIR=/app/external_lib/kenlm/

# Install kenlm
RUN apt install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
RUN apt-get install -y git
RUN mkdir external_lib && \
    cd external_lib && \
    git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=20 -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make -j 16 && \
    cd ../..

# Install Additional Dependencies (ATLAS, OpenBLAS, Accelerate, Intel MKL)
RUN apt-get install -y libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev

# Install wav2letter
RUN pip3 install packaging
RUN git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git && \
    cd wav2letter/bindings/python && \
    pip3 install -e .

RUN git clone https://github.com/pytorch/fairseq.git
RUN cd fairseq && \
    mkdir /app/data

COPY src/recognize.py /app/fairseq/examples/wav2vec/recognize.py

WORKDIR /app/fairseq
RUN pip install --editable ./ && python examples/speech_recognition/infer.py --help && python examples/wav2vec/recognize.py --help

CMD ["bash"]