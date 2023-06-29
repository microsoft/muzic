# EmoGen

EmoGen: Eliminating Subjective Bias in Emotional Music Generation, by Chenfei Kang, Peiling Lu, Botao Yu, Xu Tan, Wei Ye, Shikun Zhang, Jiang Bian, is an emotional music generation system that leverages a set of emotion-related music attributes as the bridge between emotion and music, and divides the generation into two stages: emotion-to-attribute mapping with supervised clustering, and attribute-to-music generation with self-supervised learning. Both stages are beneficial: in the first stage, the attribute values around the clustering center represent the general emotions of these samples, which help eliminate the impacts of the subjective bias of emotion labels; in the second stage, the generation is completely disentangled from emotion labels and thus free from the subjective bias. Both subjective and objective evaluations show that EMOGEN outperforms previous methods on emotion control accuracy and music quality respectively, which demonstrate our superiority in generating emotional music. 

demo: [link](https://ai-muzic.github.io/emogen/)

The following content includes the steps for EMOGEN training and inference.

### 1. Environment

- Hardware environment: We recommend Nvidia V100 16GB/32GB.

- Software environment: 

  Please make sure you have `python 3.8` installed. Run the following command to install necessary packages:

  ```sh
  bash setup.sh
  ```

​     Also, please follow the instructions in [Installation](https://jmir.sourceforge.net/manuals/jSymbolic_manual/installation_files/installation.html) to install `Java`.

### 2. Dataset

We use three datasets: One emotion-labeled dataset namely EMOPIA([link](https://annahung31.github.io/EMOPIA/))  and two unlabeled datasets namely Pop1k7 ([link](https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md)) and LMD-Piano, where LMD-Piano is con-
structed by using the samples that only contain piano tracks from the Lakh MIDI (LMD) dataset ([link](https://colinraffel.com/projects/lmd/)). To evaluate EMOGEN’s ability to generate emotional music on the arbitrary dataset, we also train EmoGen on TopMAGD([link](http://www.ifs.tuwien.ac.at/mir/msd/download.html)), which is a multi-instrument dataset. 

In principle, you can use any MIDI dataset to train the attribute-to-music generation model. Taking the 'Piano' dataset under the 'data/' folder as an example. Please follow the steps below to process the data.

1. First, put all MIDI files in 'data/Piano/midi'.

2. Run the following command to encode MIDI files.

   ```shell
   cd data_process
   python midi_encoding.py
   ```

3. Run the following command to extract attributes by jSymbolic.
  You are required to download the package jSymbolic_2_2_user.zip from https://sourceforge.net/projects/jmir/files/jSymbolic/jSymbolic%202.2/, and extract it into ./jSymbolic_lib.

   ```shell
   cd ../jSymbolic_lib
   python jSymbolic_feature.py
   ```

4. Run the following script to prepare train/validation/test dataset.

   ```shell
   cd ../data_process
   python gen_data.py
   ```

### 3.Train

- Emotion-to-attribute mapping

  In this stage, we map four emotion quadrants in Russel's 4Q model to four different attributes. We first compute attribute centers in four quadrants on the EMOPIA dataset and selected the closest attribute vector to the center in each quadrant as the mapping result. The mapped results are stored in `data/infer_input/inference_command.npy`. The emotion quadrants corresponding to these attribute vectors are shown in the following table:

  | Index                | Emotion |
  | -------------------- | ------- |
  | inference_command[0] | Q1      |
  | inference_command[1] | Q2      |
  | inference_command[2] | Q3      |
  | inference_command[3] | Q4      |

- Attribute-to-music generation

  Run the following command to train a 6-layer Linear Transformer model on the dataset `data/Piano`:

  ```shell
  bash Piano_train.sh
  ```

### 4. Inference

Please put the checkpoints under the folder `checkpoints/`. To generate piano songs, run the following command:

```shell
# usage: bash Piano_gen.sh target_emotion
#                          1-Q1 2-Q2 3-Q3 4-Q4
bash Piano_gen.sh 3 # generate songs with emotion "Q3"
```

Also, to generate multi-instrument songs, please run the following command:

```shell
bash TopMAGD_gen.sh 1 # generate songs with emotion "Q1"
```

