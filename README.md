# wav2vec
wav2vec 2.0 Recognize Implementation.

## Disclaimer
Initial issue submitted in the `fairseq` repository [here](https://github.com/pytorch/fairseq/issues/2651).

## How to install
Please install using the provided `Dockerfile`
```
docker build -t wav2vec -f Dockerfile .
docker run --rm -itd --ipc=host -v $PWD/data:/app/data --name w2v wav2vec

```

## How to Run
```
docker exec -it w2v bash
python examples/wav2vec/recognize.py --wav_path ~/data/temp.wav --w2v_path ~/data/wav2vec_small_960h.pt --target_dict_path ~/data/dict.ltr.txt
```

## Contributors
Thanks to all contributors to this repo.

- [sooftware](https://github.com/sooftware)
- [mychiux413](https://github.com/mychiux413)
