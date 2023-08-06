""" Models and APIs"""
import uuid
import numpy as np
import importlib
from transformers import pipeline, AutoConfig, Wav2Vec2FeatureExtractor
from model_utils import Wav2Vec2ForSpeechClassification, timbre_transfer
from pydub import AudioSegment
import requests
import urllib
import librosa

import torch
import torch.nn.functional as F
# import torchaudio
from fairseq.models.transformer_lm import TransformerLanguageModel
import soundfile as sf
import os
import sys
import json
import pdb


def get_task_map():
    task_map = {
        "text-to-sheet-music": [
            "sander-wood/text-to-music"
        ],
        "music-classification": [
            "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
        ],
        "lyric-to-melody": [
            "muzic/roc"
        ],
        "lyric-to-audio": [
            "DiffSinger"
        ],
        "web-search": [
            "google-search"
        ],
        "artist-search": [
            "spotify"
        ],
        "track-search": [
            "spotify"
        ],
        "album-search": [
            "spotify"
        ],
        "playlist-search": [
            "spotify"
        ],
        "separate-track": [
            "demucs"
        ],
        "lyric-recognition": [
            "jonatasgrosman/whisper-large-zh-cv11"
        ],
        "score-transcription": [
            "basic-pitch"
        ],
        "timbre-transfer": [
            "ddsp"
        ],
        "accompaniment": [
            "getmusic"
        ],
        "audio-mixing": [
            "basic-merge"
        ],
        "audio-crop": [
            "basic-crop"
        ],
        "audio-splice": [
            "basic-splice"
        ],
        "web-search": [
            "google-search"
        ],
    }

    return task_map


def init_plugins(config):
    if config["disabled_tools"] is not None:
        disabled = [tool.strip() for tool in config["disabled_tools"].split(",")]
    else: 
        disabled = []

    pipes = {}
    if "muzic/roc" not in disabled:
        pipes["muzic/roc"] = MuzicROC(config)
    if "DiffSinger" not in disabled:
        pipes["DiffSinger"] = DiffSinger(config)
    if "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres" not in disabled:
        pipes["m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"] = Wav2Vec2Base(config)
    if "jonatasgrosman/whisper-large-zh-cv11" not in disabled:
        pipes["jonatasgrosman/whisper-large-zh-cv11"] = WhisperZh(config)
    if "spotify" not in disabled:
        pipes["spotify"] = Spotify(config)
    if "ddsp" not in disabled:
        pipes["ddsp"] = DDSP(config)
    if "demucs" not in disabled:
        pipes["demucs"] = Demucs(config)
    if "basic-merge" not in disabled:
        pipes["basic-merge"] = BasicMerge(config)
    if "basic-crop" not in disabled:
        pipes["basic-crop"] = BasicCrop(config)
    if "basic-splice" not in disabled:
        pipes["basic-splice"] = BasicSplice(config)
    if "basic-pitch" not in disabled:
        pipes["basic-pitch"] = BasicPitch(config)
    if "google-search" not in disabled:
        pipes["google-search"] = GoogleSearch(config)

    return pipes


class BaseToolkit:
    def __init__(self, config):
        self.local_fold = config["local_fold"]
        self.id = "basic toolkit"
        self.attributes = {}

    def get_attributes(self):
        return json.dumps(self.attributes)

    def update_attributes(self, **kwargs):
        for key in kwargs:
            self.attributes[key] = kwargs[key]


class MuzicROC(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "muzic/roc"
        self.attributes = {
            "description": "ROC is a new paradigm for lyric-to-melody generation"
        }
        self._init_toolkit(config)

    def _init_toolkit(self, config):
        sys.path.append(os.path.join(os.getcwd(), f"{self.local_fold}/muzic/roc"))
        from main import main as roc_processer
        self.processer = roc_processer
        self.model = TransformerLanguageModel.from_pretrained(os.path.join(os.getcwd(), f"{self.local_fold}/muzic/roc/music-ckps/"), "checkpoint_best.pt", tokenizer="space",
                            batch_size=8192)
        sys.path.remove(os.path.join(os.getcwd(), f"{self.local_fold}/muzic/roc"))

    def inference(self, args, task, device="cpu"):
        results = []
        self.model.to(device)

        for arg in args: 
            if "lyric" in arg: 
                prompt = arg["lyric"]
                prompt = " ".join(prompt)

                file_name = str(uuid.uuid4())[:4]

                outputs = self.processer(
                            self.model, 
                            [prompt], 
                            output_file_name=f"public/audios/{file_name}", 
                            db_path=f"{self.local_fold}/muzic/roc/database/ROC.db"
                        )
                os.system(f"fluidsynth -l -ni -a file -z 2048 -F public/audios/{file_name}.wav 'MS Basic.sf3' public/audios/{file_name}.mid")
                results.append(
                    {
                        "score": str(outputs),
                        "audio": f"{file_name}.wav",
                        "sheet_music": f"{file_name}.mid"
                    }
                )
                

        self.model.to("cpu")
        return results


class DiffSinger(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "DiffSinger"
        self.attributes = {
            "description": "Singing Voice Synthesis via Shallow Diffusion Mechanism",
            "star": 3496
        }
        self._init_toolkit(config)

    def _init_toolkit(self, config):
        sys.path.append(os.path.join(os.getcwd(), f"{self.local_fold}/DiffSinger"))
        import utils
        importlib.reload(utils)
        from inference.svs.ds_e2e import DiffSingerE2EInfer
        from utils.hparams import hparams, set_hparams
        from utils.audio import save_wav
        work_dir = os.getcwd()
        os.chdir(os.path.join(os.getcwd(), f"{self.local_fold}/DiffSinger"))
        set_hparams('usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml', 
                    '0228_opencpop_ds100_rel', print_hparams=False)
        self.processer = save_wav
        self.model = DiffSingerE2EInfer(hparams, device="cuda:0")
        self.model.model.to("cpu")
        self.model.vocoder.to("cpu")
        os.chdir(work_dir)
        sys.path.remove(os.path.join(os.getcwd(), f"{self.local_fold}/DiffSinger"))

    def inference(self, args, task, device="cpu"):
        results = []
        self.model.model.to(device)
        self.model.vocoder.to(device)
        self.model.device = device

        for arg in args: 
            if "score" in arg:
                prompt = arg["score"]
                prompt = eval(prompt)

                wav = self.model.infer_once(prompt)
                file_name = str(uuid.uuid4())[:4]
                self.processer(wav, f"public/audios/{file_name}.wav", sr=16000)
            
                results.append({"audio": f"{file_name}.wav"})
        
        self.model.model.to("cpu")
        self.model.vocoder.to("cpu")
        self.model.device = "cpu"
        return results
    

class Wav2Vec2Base(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
        self.attributes = {
            "description": "Music Genre Classification using Wav2Vec 2.0"
        }
        self._init_toolkit(config)

    def _init_toolkit(self, config):
        self.processer = Wav2Vec2FeatureExtractor.from_pretrained(f"{self.local_fold}/m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres")
        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(f"{self.local_fold}/m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres")
        self.config = AutoConfig.from_pretrained(f"{self.local_fold}/m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres")

    def inference(self, args, task, device="cpu"):
        results = []
        self.model.to(device)

        for arg in args:
            if "audio" in arg:
                prompt = arg["audio"]

                sampling_rate = self.processer.sampling_rate
                #speech_array, _sampling_rate = torchaudio.load(prompt)
                #resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
                #speech = resampler(speech_array).squeeze().numpy()
                speech, _ = librosa.load(prompt, sr=sampling_rate)
                inputs = self.processer(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                genre = self.config.id2label[np.argmax(scores)]
                # outputs = [{"Label": pipes[pipe_id]["config"].id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
                results.append({"genre": genre})

        self.model.to("cpu")
        return results
    

class WhisperZh(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "jonatasgrosman/whisper-large-zh-cv11"
        self.attributes = {
            "description": "a fine-tuned version of openai/whisper-large-v2 on Chinese (Mandarin)"
        }
        self._init_toolkit(config)

    def _init_toolkit(self, config):
        self.model = pipeline("automatic-speech-recognition", model=f"{self.local_fold}/jonatasgrosman/whisper-large-zh-cv11", device="cuda:0")
        self.model.model.to("cpu")

    def inference(self, args, task, device="cpu"):
        results = []
        self.model.model.to(device)

        for arg in args:
            if "audio" in arg:
                prompt = arg["audio"]

                self.model.model.config.forced_decoder_ids = (
                    self.model.tokenizer.get_decoder_prompt_ids(
                        language="zh", 
                        task="transcribe"
                    )
                )
                results.append({"lyric": self.model(prompt)})
        
        self.model.model.to("cpu")
        return results
    

class Spotify(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "spotify"
        self.attributes = {
            "description": "Spotify is a digital music service that gives you access to millions of songs."
        }
        self._init_toolkit(config)

    def _init_toolkit(self, config):
        client_id = config["spotify"]["client_id"]
        client_secret = config["spotify"]["client_secret"]
        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }

        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data["access_token"]
            print("Access Token:", self.access_token)
        else:
            print("Error:", response.status_code)

    def inference(self, args, task, device="cpu"):
        results = []

        for arg in args:
            tgt = task.split("-")[0]
            
            query = ["remaster"]
            for key in arg:
                if key in ["track", "album", "artist", "genre"]:
                    if isinstance(arg[key], list):
                        value = " ".join(arg[key])
                    else: 
                        value = arg[key]
                    query.append(f"{key}:{value}")

            if tgt == "playlist":
                query[0] = arg["description"]

            query = " ".join(query).replace(" ", "%20")
            query = urllib.parse.quote(query)
            url = f"https://api.spotify.com/v1/search?query={query}&type={tgt}"
            headers = {"Authorization": f"Bearer {self.access_token}"}

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()[tgt + "s"]["items"][0]
                text = dict()
                spotify_id = data["id"]
                text[tgt] = data["name"]

                if tgt == "track":
                    if "preview_url" in data and len(data["preview_url"]) > 0:
                        url = data["preview_url"]
                        file_name = str(uuid.uuid4())[:4]
                        with open(f"public/audios/{file_name}.mp3", "wb") as f:
                            f.write(requests.get(url).content)
                        text["audio"] = f"{file_name}.mp3"

                    text["album"] = data["album"]["name"]
                    text["artist"] = [d["name"] for d in data["artists"]]
                
                if tgt == "album":
                    text["date"] = data["release_date"]
                    text["artist"] = [d["name"] for d in data["artists"]]
                    url = f"https://api.spotify.com/v1/albums/{spotify_id}"
                    album = requests.get(url, headers=headers).json()
                    
                    if len(album["genres"]) > 0:
                        text["genre"] = album["genres"]

                    text["track"] = [d["name"] for d in album["tracks"]["items"]]

                if tgt == "playlist":
                    url = f"https://api.spotify.com/v1/playlists/{spotify_id}"
                    album = requests.get(url, headers=headers).json()

                    text["track"] = [d["track"]["name"] for d in album["tracks"]["items"]]

                if tgt == "artist":
                    if len(data["genres"]) > 0:
                        text["genre"] = data["genres"]
                    
                results.append(text)

            else:
                results.append({"error": "No corresponding song found."})

        return results


class GoogleSearch(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "google"
        self.attributes = {
            "description": "Google Custom Search Engine."
        }
        self._init_toolkit(config)

    def _init_toolkit(self, config):
        api_key = config["google"]["api_key"]
        custom_search_engine_id = config["google"]["custom_search_engine_id"]
        self.url = "https://www.googleapis.com/customsearch/v1"
        self.params = {
            "key": api_key,
            "cx": custom_search_engine_id,
            "max_results": 5
        }

    def inference(self, args, task, device="cpu"):
        results = []

        for arg in args:
            if "description" in arg:
                self.params["q"] = arg["description"]
                response = requests.get(self.url, self.params)

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items")

                    descriptions = []
                    for item in items:
                        descriptions.append(
                            {
                                "title": item.get("title"),
                                "snippet": item.get("snippet")
                            }
                        )

                    results.append(
                        {
                            "description": json.dumps(descriptions)
                        }
                    )

        return results


class Demucs(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "demucs"
        self.attributes = {
            "description": "Demucs Music Source Separation"
        }

    def inference(self, args, task, device="cpu"):
        results = []
        for arg in args:
            if "audio" in arg:
                prompt = arg["audio"]

                file_name = str(uuid.uuid4())[:4]
                os.system(f"python -m demucs --two-stems=vocals -o public/audios/ {prompt}")
                os.system(f"cp public/audios/htdemucs/{prompt.split('/')[-1].split('.')[0]}/no_vocals.wav public/audios/{file_name}.wav")

                results.append({"audio": f"{file_name}.wav"})

        return results
    

class BasicMerge(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "pad-wave-merge"
        self.attributes = {
            "description": "Merge audios."
        }

    def inference(self, args, task, device="cpu"):
        audios = []
        sr=16000

        for arg in args:
            if "audio" in arg:
                audio, _ = librosa.load(arg["audio"], sr=sr)
                audios.append(audio)

        max_len = max([len(audio) for audio in audios])
        audios = [librosa.util.fix_length(audio, size=max_len) for audio in audios]

        mixed_audio = sum(audios)
        file_name = str(uuid.uuid4())[:4]

        sf.write(f"public/audios/{file_name}.wav", mixed_audio, sr)

        results = [{"audio": f"{file_name}.wav"}]

        return results
    

class DDSP(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "ddsp"
        self.attributes = {
            "description": "Convert audio between sound sources with pretrained models."
        }

    def inference(self, args, task, device="cpu"):
        results = []

        for arg in args:
            if "audio" in arg:
                prompt = arg["audio"]
                file_name = str(uuid.uuid4())[:4]
                timbre_transfer(prompt, f"public/audios/{file_name}.wav", instrument="violin")
                results.append({"audio": f"{file_name}.wav"})

        return results


class BasicPitch(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "basic-pitch"
        self.attributes = {
            "description": "Demucs Music Source Separation"
        }

    def inference(self, args, task, device="cpu"):
        results = []
        for arg in args:
            if "audio" in arg:
                prompt = arg["audio"]

                file_name = str(uuid.uuid4())[:4]
                os.system(f"basic-pitch public/audios/ {prompt}")
                os.system(f"cp public/audios/{prompt.split('/')[-1].split('.')[0]}_basic_pitch.mid public/audios/{file_name}.mid")

                results.append({"sheet music": f"{file_name}.mid"})

        return results


class BasicCrop(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "audio-crop"
        self.attributes = {
            "description": "Trim audio based on time"
        }

    def inference(self, args, task, device="cpu"):
        results = []
        for arg in args:
            if "audio" in arg and "time" in arg:
                prompt = arg["audio"]
                time = arg["time"]

                file_name = str(uuid.uuid4())[:4]
                audio = AudioSegment.from_file(prompt)
                start_ms = int(float(time[0]) * 1000)
                end_ms = int(float(time[1]) * 1000)

                if start_ms < 0:
                    start_ms += len(audio)
                if end_ms < 0:
                    end_ms += len(audio)

                start_ms = max(start_ms, len(audio))
                end_ms = max(end_ms, len(audio))

                if start_ms > end_ms:
                    continue

                trimmed_audio = audio[start_ms:end_ms]
                trimmed_audio.export(f"public/audios/{file_name}.wav", format="wav")
                results.append({"audio": f"{file_name}.wav"})

        return results
    

class BasicSplice(BaseToolkit):
    def __init__(self, config):
        super().__init__(config)
        self.id = "audio-splice"
        self.attributes = {
            "description": "Basic audio splice"
        }

    def inference(self, args, task, device="cpu"):
        audios = []
        results = []

        for arg in args:
            if "audio" in arg:
                audios.append(arg["audio"])

        audio = AudioSegment.from_file(audios[0])
        for i in range(1, len(audios)):
            audio = audio + AudioSegment.from_file(audios[i])

        file_name = str(uuid.uuid4())[:4]
        audio.export(f"public/audios/{file_name}.wav", format="wav")
        results.append({"audio": f"{file_name}.wav"})

        return results
