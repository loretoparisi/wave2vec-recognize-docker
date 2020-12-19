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
We make use of `python:3.8.6-slim-buster` as base image in order to let developers to have more flexibility in customize this `Dockerfile`. For a simplifed install please refer to [Alternative Install](#Alternative-Install) section. If you go for this container, please install using the provided `Dockerfile`
```bash
docker build -t wav2vec -f Dockerfile .
```

## How to Run
There are two version of `recognize.py`.
- `recognize.py`: For running legacy finetuned model (without Hydra).
- `recognize.hydra.py`: For running new finetuned with newer version of **fairseq**.

Before running, please copy the downloaded model (e.g. `wav2vec_small_10m.pt`) to the `data/` folder. Please copy there the wav file to test as well, like `data/temp.wav` in the following examples.  So the `data/` folder will now look like this

```
.
├── dict.ltr.txt
├── temp.wav
└── wav2vec_small_10m.pt
```

We now run the container and the we enter and execute the recognition (`recognize.py` or `recognize.hydra.py`).
```bash
docker run -d -it --rm -v $PWD/data:/app/data --name w2v wav2vec
docker exec -it w2v bash
python examples/wav2vec/recognize.py --target_dict_path=/app/data/dict.ltr.txt /app/data/wav2vec_small_10m.pt /app/data/temp.wav
```

## Common issues
### 1. What if my model are not compatible with **fairseq**?

At the very least, we have tested with fairseq master branch (> v0.10.1, commit [ac11107](https://github.com/pytorch/fairseq/commit/ac11107ed41cb06a758af850373c239309d1c961)). When you run into issues, like this:
```txt
omegaconf.errors.ValidationError: Invalid value 'False', expected one of [hard, soft]
full_key: generation.print_alignment
reference_type=GenerationConfig
object_type=GenerationConfig
```
It's probably that your model've been finetuned (or trained) with other version of **fairseq**.
You should find yourself which version your model are trained, and edit commit hash in Dockerfile accordingly, **BUT IT MIGHT BREAK src/recognize.py**.

The workaround is look for what's changed in the parameters inside **fairseq** source code. In the above example, I've managed to find that:

***fairseq/dataclass/configs.py (72a25a4 -> 032a404)***
```diff
- print_alignment: bool = field(
+ print_alignment: Optional[PRINT_ALIGNMENT_CHOICES] = field(
-     default=False,
+     default=None,
      metadata={
-         "help": "if set, uses attention feedback to compute and print alignment to source tokens"
+         "help": "if set, uses attention feedback to compute and print alignment to source tokens "
+         "(valid options are: hard, soft, otherwise treated as hard alignment)",
+         "argparse_const": "hard",
      },
  )
```
The problem is fairseq had modified such that `generation.print_alignment` not valid anymore, so I modify `recognize.hydra.py` as below (you might wanna modify the value instead):
```diff
  OmegaConf.set_struct(w2v["cfg"], False)
+ del w2v["cfg"].generation["print_alignment"]
  cfg = OmegaConf.merge(OmegaConf.structured(Wav2Vec2CheckpointConfig), w2v["cfg"])
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
