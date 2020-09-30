# wav2vec
wav2vec 2.0 Recognize Implementation. 

## Disclaimer
[Wave2vec](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) is part of [fairseq](https://github.com/pytorch/fairseq)
This repository is the result of the issue submitted in the `fairseq` repository [here](https://github.com/pytorch/fairseq/issues/2651).

## Resource
Please first download one of the pre-trained models available from `fairseq` (see later).

## Pre-trained models

Model | Finetuning split | Dataset | Model
|---|---|---|---
Wav2Vec 2.0 Base | No finetuning | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt)
Wav2Vec 2.0 Base | 10 minutes | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_10m.pt)
Wav2Vec 2.0 Base | 100 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_100h.pt)
Wav2Vec 2.0 Base | 960 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt)
Wav2Vec 2.0 Large | No finetuning | [Librispeech](http://www.openslr.org/12)  | [download](https//dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt)
Wav2Vec 2.0 Large | 10 minutes | [Librispeech](http://www.openslr.org/12)  | [download](https//dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_10m.pt)
Wav2Vec 2.0 Large | 100 hours | [Librispeech](http://www.openslr.org/12)  | [download](https//dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_100h.pt)
Wav2Vec 2.0 Large | 960 hours | [Librispeech](http://www.openslr.org/12)  | [download](https//dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_960h.pt)
Wav2Vec 2.0 Large (LV-60) | No finetuning | [Libri-Light](https://github.com/facebookresearch/libri-light) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox.pt)
Wav2Vec 2.0 Large (LV-60) | 10 minutes | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_10m.pt)
Wav2Vec 2.0 Large (LV-60) | 100 hours | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_100h.pt)
Wav2Vec 2.0 Large (LV-60) | 960 hours | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec2_vox_960h.pt)


## How to install
We make use of `python:3.7.4-slim-buster` as base image in order to let developers to have more flexibility in customize this `Dockerfile`. For a simplifed install please refer to [Alternative Install](#Alternative-Install) section. If you go for this container, please install using the provided `Dockerfile`
```bash
docker build -t wav2vec -f Dockerfile .
```

## How to Run
Before running, please copy the downloaded model (e.g. `wav2vec_small_10m.pt`) to the `data/` folder. Please copy there the wav file to test as well, like `data/temp.wav` in the following examples.  So the `data/` folder will now look like this

```
.
├── dict.ltr.txt
├── temp.wav
└── wav2vec_small_10m.pt
```

We now run the container as a daemon and the we enter and execute the recognition.
```bash
docker run -d -it --rm -v $PWD/data:/app/data --name w2v wav2vec
docker exec -it w2v bash
python examples/wav2vec/recognize.py --wav_path /app/data/temp.wav --w2v_path /app/data/wav2vec_small_10m.pt --target_dict_path /app/data/dict.ltr.txt 
```

## Alternative install
We provide an alternative Dockerfile named `wav2letter.Dockerfile` that makes use of `wav2letter/wav2letter:cpu-latest` Docker image as `FROM`.
Here are the commands for build, install and run in this case:

```bash
docker build -t wav2vec2 -f wav2letter.Dockerfile .
docker run -d -it --rm -v $PWD/data:/root/data --name w2v2 wav2vec2
docker exec -it w2v2 bash
python examples/wav2vec/recognize.py --wav_path /root/data/temp.wav --w2v_path /root/data/wav2vec_small_10m.pt --target_dict_path /root/data/dict.ltr.txt 
```

## Contributors
Thanks to all contributors to this repo.

- [sooftware](https://github.com/sooftware)
- [mychiux413](https://github.com/mychiux413)
