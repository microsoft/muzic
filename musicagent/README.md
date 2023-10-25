# Music Agent 

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2310.11954)
[![Open in Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue)]()

Music Agent stands as an LLM-powered autonomous agent within the realm of music. Its modular and highly extensible framework liberates you to focus on the most imaginative aspects of music comprehension and composition!

## Demo Video

[![Watch the video](https://img.youtube.com/vi/tpNynjdcBqA/maxresdefault.jpg)](https://youtu.be/tpNynjdcBqA)

## Features

- Accessibility: Music Agent dynamically selects the most appropriate methods for each music-related task.
- Unity: Music Agent unifies a wide array of tools into a single system, incorporating Huggingface models, GitHub projects, and Web APIs.
- Modularity: Music Agent offers high modularity, allowing users to effortlessly enhance its capabilities by integrating new functions.

## Installation

### Docker (Recommended)

To be created.

### Conda / Pip

#### Install Dependencies

To set up the system from source, follow the steps below:

```bash
# Make sure git-lfs is installed
sudo apt-get update
sudo apt-get install -y git-lfs

# Install music-related libs
sudo apt-get install -y libsndfile1-dev
sudo apt-get install -y fluidsynth
sudo apt-get install -y ffmpeg
sudo apt-get install -y lilypond

# Clone the repository from muzic
git clone https://github.com/muzic
cd muzic/agent
```

Next, install the dependent libraries. There might be some conflicts, but they should not affect the functionality of the system.

```bash
pip install --upgrade pip

pip install semantic-kernel
pip install -r requirements.txt
pip install numpy==1.23.0
pip install protobuf==3.20.3
```

By following these steps, you will be able to successfully set up the system from the provided source.

#### Download Huggingface / Github Parameters

```bash
cd models/  # Or your custom folder for tools
bash download.sh
```

P.S. Download Github parameters according to your own need: 

To use [muzic/roc](https://github.com/microsoft/muzic/tree/main/roc), follow these steps:

```bash
cd YOUR_MODEL_DIR   # models/ by default
cd muzic/roc
```

1. Download the checkpoint and database from the following [link](https://drive.google.com/drive/folders/1TpWOMlRAaUL-R6CRLWfZK1ZeE1VCaubp).
2. Place the downloaded checkpoint file in the *music-ckpt* folder.
3. Create a folder named *database* to store the downloaded database files.

To use [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger), follow these steps:

```bash
cd YOUR_MODEL_DIR
cd DiffSinger
```

1. Down the checkpoint and config from the following [link](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0228_opencpop_ds100_rel.zip) and unzip it in *checkpoints* folder.
2. You can find other DiffSinger checkpoints in its [docs](https://github.com/MoonInTheRiver/DiffSinger/blob/master/docs/README-SVS.md)

To use [DDSP](https://github.com/magenta/ddsp/tree/main), follow these steps:

```bash
cd YOUR_MODEL_DIR
mkdir ddsp
cd ddsp

pip install gsutil
mkdir violin; gsutil cp gs://ddsp/models/timbre_transfer_colab/2021-07-08/solo_violin_ckpt/* violin/
mkdir flute; gsutil cp gs://ddsp/models/timbre_transfer_colab/2021-07-08/solo_flute_ckpt/* flute/
```

To use audio synthesis, please download [MS Basic.sf3](https://github.com/musescore/MuseScore/tree/master/share/sound) and place it in the main folder.

## Usage

Change the *config.yaml* file to ensure that it is suitable for your application scenario.

```yaml
# optional tools
huggingface:
  token: YOUR_HF_TOKEN
spotify:
  client_id: YOUR_CLIENT_ID
  client_secret: YOUR_CLIENT_SECRET
google:
  api_key: YOUR_API_KEY
  custom_search_engine_id: YOUR_SEARCH_ENGINE_ID
```

- Set your [Hugging Face token](https://huggingface.co/settings/tokens).
- Set your [Spotify Client ID and Secret](https://developer.spotify.com/dashboard), according to the [doc](https://developer.spotify.com/documentation/web-api).
- Set your [Google API key](https://console.cloud.google.com/apis/dashboard) and [Google Custom Search Engine ID](https://programmablesearchengine.google.com/controlpanel/create)

### CLI

fill the .env

```bash
OPENAI_API_KEY=""
OPENAI_ORG_ID=""

# optional
AZURE_OPENAI_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
```

If you use Azure OpenAI, please pay attention to change *use_azure_openai* in *config.yaml*.

And now you can run the agent by:

```bash
python agent.py --config config.yaml
```

### Gradio

We also provide gradio interface

```bash
python gradio_agent.py --config config.yaml
```

No .env file setup is required for Gradio interaction selection, but it does support only the OpenAI key.


## Citation

If you use this code, please cite it as:

```
@article{yu2023musicagent,
  title={MusicAgent: An AI Agent for Music Understanding and Generation with Large Language Models},
  author={Yu, Dingyao and Song, Kaitao and Lu, Peiling and He, Tianyu and Tan, Xu and Ye, Wei and Zhang, Shikun and Bian, Jiang},
  journal={arXiv preprint arXiv:2310.11954},
  year={2023}
}
```