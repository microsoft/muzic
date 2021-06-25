# MusicBERT

* The paper: [MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training](https://arxiv.org/pdf/2106.05630.pdf)

## Preparing environment for MusicBERT

* Download Anaconda install script and install it on current directory

  ```bash
  wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
  bash Anaconda3-2020.07-Linux-x86_64.sh -b -p anaconda3
  anaconda3/bin/conda create --name musicbert python=3.7 -y
  anaconda3/bin/activate musicbert
  conda install pytorch=1.4.0 cudatoolkit=10.0 -c pytorch -y
  pip install sklearn miditoolkit matplotlib
  ```

* Install fairseq (version 336942734c85791a90baa373c212d27e7c722662)

  ```bash
  git clone https://github.com/pytorch/fairseq
  cd fairseq
  git checkout 336942734c85791a90baa373c212d27e7c722662
  pip install --editable ./
  ```

* Install apex for faster training (optional)

## Preparing dataset for pre-training and downstream tasks

* Patch fairseq binarizer (`fairseq/fairseq/binarizer.py`) because preprocessed data already contain eos tokens (`</s>`)

  ```python
  class Binarizer:
      @staticmethod
      def binarize(...):
          append_eos = False  # add this line to always disable append_eos functionality of binarizer
  ```

### Pre-training

* Prepare a zip of midi files for pre-training (say `manymidi.zip`)

* Run the dataset processing script (`preprocess.py`)

  ```bash
  python -u preprocess.py
  ```

* The script should prompt you to input the path of the midi zip and the path for OctupleMIDI output

  ```
  Dataset zip path: /xxx/xxx/MusicBERT/manymidi.zip
  OctupleMIDI output path: manymidi_data_raw
  SUCCESS: manymidi/a/0000.mid
  SUCCESS: manymidi/b/0001.mid
  SUCCESS: manymidi/c/0002.mid
  SUCCESS: manymidi/d/0005.mid
  ......
  ```

* Binarize the raw text format dataset (this script will read `manymidi_data_raw` folder and output `manymidi_data_bin`)

  ```bash
  bash binarize_pretrain.sh manymidi
  ```

### Melody completion task and accompaniment suggestion task

* Follow "PiRhDy: Learning Pitch-, Rhythm-, and Dynamics-aware Embeddings for Symbolic Music" (ACM MM 2020 BEST PAPER) (https://github.com/mengshor/PiRhDy) to generate datasets for melody completion task and accompaniment suggestion task

  ```
  PiRhDy/dataset/context_next/train
  PiRhDy/dataset/context_next/test
  PiRhDy/dataset/context_acc/train
  PiRhDy/dataset/context_acc/test
  ```

* Convert these two datasets to OctupleMIDI format with `gen_nsp.py`

  ```bash
  python -u gen_nsp.py
  ```

* The script should prompt you to input which downstream task to process (next for melody task and acc for accompaniment task)

  ```
  task = next
  ```

  ```
  task = acc
  ```

### Genre and style classification task

* Prepare [The Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (**LMD-full**) in zip format (say `lmd_full.zip`)

  ```
  wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
  tar -xzvf lmd_full.tar.gz
  zip -r lmd_full.zip lmd_full
  ```

* Get TOPMAGD and MASD midi to genre mapping [midi_genre_map](https://github.com/andrebola/patterns-genres/blob/master/data/midi_genre_map.json) from "On large-scale genre classification in symbolically encoded music by automatic identification of repeating patterns" (DLfM 2018) (https://github.com/andrebola/patterns-genres)

  ```
  wget https://raw.githubusercontent.com/andrebola/patterns-genres/master/data/midi_genre_map.json
  ```

* Generate these two datasets in OctupleMIDI format using the midi to genre mapping file with `gen_genre.py`

  ```bash
  python -u gen_genre.py
  ```

* The script should prompt you to input which downstream task to process (topmagd for genre task and masd for style task)

  ```bash
  subset: topmagd
  LMD dataset zip path: lmd_full.zip
  sequence length: 1000
  ```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.