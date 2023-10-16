#!/bin/bash

# Set models to download
models=(
    "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
	"lewtun/distilhubert-finetuned-music-genres"
	"dima806/music_genres_classification"
    "sander-wood/text-to-music"
    "jonatasgrosman/whisper-large-zh-cv11"
	"cvssp/audioldm-m-full"
)

# Set the current directory
CURRENT_DIR=$(pwd)

# Download models
for model in "${models[@]}"; do
	echo "----- Downloading from https://huggingface.co/${model} -----"
	if [ -d "${model}" ]; then
		(cd "${model}" && git pull && git lfs pull)
	else
		git clone --recurse-submodules "https://huggingface.co/${model}" "${model}"
	fi
done

# Set Git project to download
libs=(
	"microsoft/muzic"
	"MoonInTheRiver/DiffSinger"
)
for lib in "${libs[@]}"; do
	echo "----- Downloading from https://github.com/${lib}.git -----"
	git clone "https://github.com/${lib}.git"
done

cp -r ../auxiliary/* ./
