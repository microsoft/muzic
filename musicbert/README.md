# MusicBERT

* The paper: [MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training](https://arxiv.org/pdf/2106.05630.pdf)

## Preparing datasets

### Pre-training

* Prepare [The Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (**LMD-full**) in zip format for pre-training. (say `lmd_full.zip`)

  ```bash
  wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
  tar -xzvf lmd_full.tar.gz
  zip -r lmd_full.zip lmd_full
  ```

* Run the dataset processing script. (`preprocess.py`)

  ```bash
  python -u preprocess.py
  ```

* The script should prompt you to input the path of the midi zip and the path for OctupleMIDI output.

  ```
  Dataset zip path: /xxx/xxx/MusicBERT/lmd_full.zip
  OctupleMIDI output path: lmd_full_data_raw
  SUCCESS: lmd_full/a/0000.mid
  ......
  ```
  
* Binarize the raw text format dataset. (this script will read `lmd_full_data_raw` folder and output `lmd_full_data_bin`)

  ```bash
  bash binarize_pretrain.sh lmd_full
  ```
  
* Pre-training checkpoints are available in Releases "MusicBert pretrain checkpoints-v1.0".  You should save in the ` checkpoints` folder.

### Melody completion task and accompaniment suggestion task

* Follow "PiRhDy: Learning Pitch-, Rhythm-, and Dynamics-aware Embeddings for Symbolic Music" (ACM MM 2020 BEST PAPER) (https://github.com/mengshor/PiRhDy) to generate datasets for melody completion task and accompaniment suggestion task.

  ```
  PiRhDy/dataset/context_next/train
  PiRhDy/dataset/context_next/test
  PiRhDy/dataset/context_acc/train
  PiRhDy/dataset/context_acc/test
  ```

* Convert these two datasets to OctupleMIDI format with `gen_nsp.py`.

  ```bash
  python -u gen_nsp.py
  ```

* The script should prompt you to input which downstream task to process. (**next** for melody task and **acc** for accompaniment task)

  ```
  task = next
  ```

* Binarize the raw text format dataset. (this script will read `next_data_raw` folder and output `next_data_bin`)

  ```bash
  bash binarize_nsp.sh next
  ```

### Genre and style classification task

* Prepare [The Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (**LMD-full**) in zip format. (say `lmd_full.zip`,  ignore this step if you have `lmd_full.zip`)

  ```bash
  wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
  tar -xzvf lmd_full.tar.gz
  zip -r lmd_full.zip lmd_full
  ```

* Get TOPMAGD and MASD midi to genre mapping [midi_genre_map](https://github.com/andrebola/patterns-genres/blob/master/data/midi_genre_map.json) from "On large-scale genre classification in symbolically encoded music by automatic identification of repeating patterns".(DLfM 2018) (https://github.com/andrebola/patterns-genres)

  ```bash
  wget https://raw.githubusercontent.com/andrebola/patterns-genres/master/data/midi_genre_map.json
  ```

* Generate these two datasets in OctupleMIDI format using the midi to genre mapping file with `gen_genre.py`.

  ```bash
  python -u gen_genre.py
  ```

* The script should prompt you to input which downstream task to process. (**topmagd** for genre task and **masd** for style task)

  ```bash
  subset: topmagd
  LMD dataset zip path: lmd_full.zip
  sequence length: 1000
  ```

* Binarize the raw text format dataset. (this script will read `topmagd_data_raw` folder and output `topmagd_data_bin`)

  ```bash
  bash binarize_genre.sh topmagd
  ```

## Training

### Pre-training

```bash
bash train_mask.sh
```

### Melody completion task and accompaniment suggestion task

```bash
bash train_nsp.sh next checkpoints/checkpoint_last_musicbert_base.pt
```

```bash
bash train_nsp.sh acc checkpoints/checkpoint_last_musicbert_small.pt
```

### Genre and style classification task

```bash
bash train_genre.sh topmagd 13 0 checkpoints/checkpoint_last_musicbert_base.pt
```

```bash
bash train_genre.sh masd 25 4 checkpoints/checkpoint_last_musicbert_small.pt
```

## Evaluation

### Melody completion task and accompaniment suggestion task

```bash
python -u eval_nsp.py checkpoints/checkpoint_last_nsp_next_checkpoint_last_musicbert_base.pt next_data_bin
```

### Genre and style classification task

```bash
python -u eval_genre.py checkpoints/checkpoint_last_genre_topmagd_x_checkpoint_last_musicbert_small.pt topmagd_data_bin/x
```

