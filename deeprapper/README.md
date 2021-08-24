# DeepRapper

* The paper: [DeepRapper: Neural Rap Generation with Rhyme and Rhythm Modeling](https://arxiv.org/abs/2107.01875)


## Data Preparation
Prepare both lyrics and pinyin for each song. We provide some data samples in `DeepRapper/data/`.

```bash
├── data
│   └── lyrics
│       └── lyrics_samples
│           └── raw
│               └── singer01
│                   └── album01
│                       ├── song01
│                       │   ├── lyric_with_beat_global.txt
│                       │   └── mapped_final_with_beat_global.txt
│                       └── song02
│                           ├── lyric_with_beat_global.txt
│                           └── mapped_final_with_beat_global.txt
```

Here is a sample of `lyric_with_beat_global.txt`:
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
Here is a sample of `mapped_final_with_beat_global.txt`:
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
When training, you may see the logs:

```bash
starting training
epoch 1
time: 2021-xx-xx 11:17:57.067011
51200
now time: 11:17. Step 10 of piece 0 of epoch 1, loss 9.587631130218506
now time: 11:18. Step 20 of piece 0 of epoch 1, loss 9.187388515472412
```
You can specify the arguments in the bash file, such as number of epoch, bach size, etc.

To generate, run:

```bash
bash generate.sh
```
You can specify the arguments in the bash file, such as beam width, number of samples, etc.

For more generated samples, visit https://deeprapper.github.io.

## Pretrained Model
The pretained model can be found in `Releases: DeepRapper-v1.0`.