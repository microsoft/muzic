# MusicBERT

* [MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training](https://arxiv.org/pdf/2106.05630.pdf)

## Preparing environment for MusicBERT

* Download Anaconda install script and install it on current directory

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh -b -p anaconda3
anaconda3/bin/conda create --name musicbert python=3.7 -y
anaconda3/bin/activate musicbert
conda install pytorch=1.4.0 cudatoolkit=10.0 -c pytorch -y
pip install sklearn miditoolkit matplotlib
```

* Install fairseq (version 336942734c85791a90baa373c212d27e7c722662)

```
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout 336942734c85791a90baa373c212d27e7c722662
pip install --editable ./
```

* Install apex for faster training (optional)

## Preparing dataset for pre-training and downstream tasks

### Pre-training

* Patch fairseq binarizer (`fairseq/fairseq/binarizer.py`) because preprocessed data already contain eos tokens (`</s>`)

  ```
  class Binarizer:
      @staticmethod
      def binarize(...):
  		append_eos = False  # add this line to always disable append_eos functionality of binarizer
  ```

* Prepare a zip of midi files for pre-training (say `lmd_full.zip`)

* Run the script `preprocessing.py`

  ```
  python -u preprocessing
  ```

* The script should prompt you to input midi zip path and OctupleMIDI output path

  ```
  Dataset zip path: /xxx/xxx/MusicBERT/lmd_full.zip
  OctupleMIDI output path: lmd_data_raw
  SUCCESS: lmd_full/6/689c4bf6e5302a2f383f335c26b6ab9b.mid
  SUCCESS: lmd_full/3/3b18722086892cfe521b905f2a4f6ee0.mid
  SUCCESS: lmd_full/3/3f5b77901be8f20ffa6425bc7cad5fed.mid
  SUCCESS: lmd_full/e/eb1289f4b642c83de3c155ff7f4234c3.mid
  ......
  ```

* Binarize raw text format (this script will read lmd_data_raw folder and output lmd_data_bin)

  ```
  bash binarize_pretrain.sh lmd
  ```

### Melody completion task and accompaniment suggestion task

### Genre and style classification task

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