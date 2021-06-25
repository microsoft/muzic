# DeepRapper

* The paper: [MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training]()

## Requirements
The requirements for running *DeepRapper* are listed in `DeepRapper/requirements.txt`. To install the requirements, run:
    ```bash
    pip install -r requirements.txt 
    ```
## Data Preparation
Prepare both lyrics and pinyin for each song.

    ```bash
    ├── data
    │   └── lyrics
    │       └── rap
    │           └── singer01
    │               └── album01
    │                   ├── song01
    │                   │   ├── lyrics.txt
    │                   │   └── pinyin.txt
    │                   └── song02
    │                       ├── lyrics.txt
    │                       └── pinyin.txt
    ```
Here is a sample of `lyrics.txt`:
```
20_[01:12.56][BEAT]那就[BEAT]让我再沉[BEAT]沦这一世
21_[01:14.49][BEAT]不理[BEAT]解早已[BEAT]经不止一次
22_[01:16.59][BEAT]那就[BEAT]让我孤[BEAT]注最后一掷
23_[01:18.61][BEAT]不想昏[BEAT]暗之中[BEAT]度过每日
24_[01:20.60][BEAT]那就[BEAT]让我再[BEAT]沉沦这一世
25_[01:22.48][BEAT]不理[BEAT]解早已[BEAT]经不止一次
26_[01:24.58][BEAT]那就[BEAT]让我孤[BEAT]注最后一掷
27_[01:26.47][BEAT]不想昏[BEAT]暗之[BEAT]中度过每日
```
Here is a sample of `pinyin.txt`:
```
20_[01:12.56][BEAT] a ou [BEAT] ang o ai en [BEAT] en e i i
21_[01:14.49][BEAT] u i [BEAT] ie ao i [BEAT] in u i i i
22_[01:16.59][BEAT] a ou [BEAT] ang o u [BEAT] u ei ou i i
23_[01:18.61][BEAT] u ang en [BEAT] an i ong [BEAT] u o ei i
24_[01:20.60][BEAT] a ou [BEAT] ang o ai [BEAT] en en e i i
25_[01:22.48][BEAT] u i [BEAT] ie ao i [BEAT] in u i i i
26_[01:24.58][BEAT] a ou [BEAT] ang o u [BEAT] u ei ou i i
27_[01:26.47][BEAT] u ang en [BEAT] an i [BEAT] ong u o ei i
```

## Train & Generation
We provide a example script for train and generation.
To train, run:

    ```bash
    bash train.sh
    ```
To generate, run:

    ```bash
    bash generate.sh
    ```