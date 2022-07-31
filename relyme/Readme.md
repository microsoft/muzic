# ReLyMe

[ReLyMe: Improving Lyric-to-Melody Generation by Incorporating Lyric-Melody Relationships](https://arxiv.org/pdf/2207.05688.pdf), by Chen Zhang, LuChin Chang, Songruoyao Wu, Xu Tan, Tao Qin, Tie-Yan Liu, Kejun Zhang, ACMMM 2022, is a method that leverages lyric-melody relationships from music theory to alleviate the dissonance between lyrics and melodies. Specifically, we first introduce several principles that lyrics and melodies should follow in terms of tone, rhythm, and structure relationships according to musicians and composers. These principles are then integrated into neural network based lyric-to-melody models by adding corresponding constraints during the decoding process to improve the harmony between lyrics and melodies. We further design a series of objective and subjective metrics to evaluate the generated melodies. Experiments on both English and Chinese song datasets show the effectiveness of ReLyMe, demonstrating the superiority of leveraging the principles (lyric-melody relationships) from music domain for neural based lyric-to-melody generation.

<p align="center">
	<video width=100% height=100% controls>
  		<source src="/videos/demo_video.mp4"  type="video/mp4">
	</video>
<br/> The demo video of ReLyMe. </p>

## 1 Environment

All python packages used in ReLyMe are listed in requirements.txt
```python
pip install -r requirements.txt
```

## 2 Traning

For the training details of TeleMelody and SongMASS, please refer to [https://github.com/microsoft/muzic/tree/main/telemelody](https://github.com/microsoft/muzic/tree/main/telemelody) and [https://github.com/microsoft/muzic/tree/main/songmass](https://github.com/microsoft/muzic/tree/main/songmass).


## 3 TeleMelody

To use ReLyMe in TeleMelody, please first follow the instructions [here](https://github.com/microsoft/muzic/tree/main/telemelody), and have it run successfully on your machine. After having TeleMelody work successfully, please follow the steps below:

1. Copy all the files under ReLyMe/telemelody_zh or ReLyMe/telemelody_en to YOUR_TELEMELODY_PATH/inferrence
2. Then, replace sequence_generator.py and fairseq_task.py in Fairseq packages with the ones we provide in ReLyMe/telemelody_zh or ReLyMe/telemelody_en.
```python
cd [YOUR PYTHON INTERPRETER PATH]/lib/python3.7/site-packages/fairseq
mv sequence_generator.py sequence_generator_bk.py
mv fairseq_task.py fairseq_task_bk.py
cp sequence_generator.py [YOUR PYTHON INTERPRETER PATH]/lib/python3.7/site-packages/fairseq/
cp fairseq_task.py [YOUR PYTHON INTERPRETER PATH]/lib/python3.7/site-packages/fairseq/tasks
```
3. Set the global variable "GEN_MODE" in config.py to "BASE" for generating TeleMelody baseline output, and "ReLyMe" for ReLyMe output.
```python
import numpy as np
from enum import Enum
from midi_utils import number_to_note

GEN_MODE = "BASE"
# GEN_MODE = "ReLyMe"
```
4. Finally, run the main.py
```shell
python main.py
```

## 4 SongMASS
(TBD)

## 5 Evaluation

We provide a score module (ReLyMe/score) to perform objective evaluation. For Chinese, you should prepare {zh_song_prefix}.mid (midi files) and {zh_song_prefix}.strct (structure file) in the same directory. For English, you should prepare {en_song_prefix}.mid (midi files), {en_song_prefix}.strct (structure file) {en_song_prefix}.syl (syllable file) in the same directory.

We provide sample files under ReLyMe/score/testmid.

```shell
score/testmid
├── en
│   ├── tele-en.mid
│   ├── tele-en.strct
│   └── tele-en.syl
└── zh
    ├── tele-zh.mid
    └── tele-zh.strct
```

There is two way you can use the score module.

1. Use the score_zh.py or score_en.py directly:
   Set the song_prefix to the one you want at the final lines in score_zh.py or score_en.py. And then run score_zh.py or score_en.py.
```python
if __name__ == "__main__":
	score = get_score("testmid/zh/tele-zh")
```

```python
python score_en.py
```

2. Import the ReLyMe/score as an module:
   Import module and use the score.get_score() to get the score of song_prefix
```python
import os
import sys
sys.path.append(os.path.join({PATH_TO_ReLyMe/score}))
import score.score_en as score_en
```

```python
print(score_en.get_score(song_prefix))
```

