import os
import requests
import urllib.parse
import librosa
import soundfile as sf
import re
import numpy as np
import ddsp
import ddsp.training
import pickle
import gin
from ddsp.training.postprocessing import (
    detect_notes, fit_quantile_transform
)
# from ddsp.colab.colab_utils import (
#     auto_tune, get_tuning_factor
# )
import tensorflow.compat.v2 as tf
import pdb
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from transformers.models.hubert.modeling_hubert import (
    HubertPreTrainedModel,
    HubertModel
)

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.file_utils import ModelOutput


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def shift_ld(audio_features, ld_shift=0.0):
    """Shift loudness by a number of ocatves."""
    audio_features['loudness_db'] += ld_shift
    return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
    """Shift f0 by a number of ocatves."""
    audio_features['f0_hz'] *= 2.0 ** (pitch_shift)
    audio_features['f0_hz'] = np.clip(audio_features['f0_hz'], 
                                        0.0, 
                                        librosa.midi_to_hz(110.0))
    return audio_features


def timbre_transfer(filename, out_path, instrument="violin", sample_rate=16000):
    audio, _ = librosa.load(filename, sr=sample_rate)
    audio = audio[np.newaxis, :]

    # Setup the session.
    ddsp.spectral_ops.reset_crepe()
    audio_features = ddsp.training.metrics.compute_audio_features(audio)
    model_dir = f"models/ddsp/{instrument}"
    gin_file = os.path.join(model_dir, 'operative_config-0.gin')
    DATASET_STATS = None
    dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
    if tf.io.gfile.exists(dataset_stats_file):
        with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
            DATASET_STATS = pickle.load(f)

    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = os.path.join(model_dir, ckpt_name)
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    hop_size = int(n_samples_train / time_steps_train)
    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    gin_params = [
        'Harmonic.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
        'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features['audio'] = audio_features['audio'][:, :n_samples]

    model = ddsp.training.models.Autoencoder()
    model.restore(ckpt)

    threshold = 1
    ADJUST = True
    quiet = 20
    autotune = 0
    pitch_shift = 0
    loudness_shift = 0
    audio_features_mod = {k: v.copy() for k, v in audio_features.items()}
    mask_on = None

    if ADJUST and DATASET_STATS is not None:
        # Detect sections that are "on".
        mask_on, note_on_value = detect_notes(audio_features['loudness_db'],
                                                audio_features['f0_confidence'],
                                                threshold)
        if np.any(mask_on):
            # Shift the pitch register.
            target_mean_pitch = DATASET_STATS['mean_pitch']
            pitch = ddsp.core.hz_to_midi(audio_features['f0_hz'])
            mean_pitch = np.mean(pitch[mask_on])
            p_diff = target_mean_pitch - mean_pitch
            p_diff_octave = p_diff / 12.0
            round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
            p_diff_octave = round_fn(p_diff_octave)
            audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)
            # Quantile shift the note_on parts.
            _, loudness_norm = fit_quantile_transform(
                audio_features['loudness_db'],
                mask_on,
                inv_quantile=DATASET_STATS['quantile_transform'])
            # Turn down the note_off parts.
            mask_off = np.logical_not(mask_on)
            loudness_norm[mask_off] -=  quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
            loudness_norm = np.reshape(loudness_norm, audio_features['loudness_db'].shape)
            audio_features_mod['loudness_db'] = loudness_norm 
            # Auto-tune.
            if autotune:
                f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod['f0_hz']))
                tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
                f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=autotune)
                audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)
        else:
            print('\nSkipping auto-adjust (no notes detected or ADJUST box empty).')

    af = audio_features if audio_features_mod is None else audio_features_mod
    outputs = model(af, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    sf.write(out_path, audio_gen[0], sample_rate)


def pad_wave_mixing(file_name1, file_name2, out_path='mixed_audio.wav', sr=16000):
    audio1, _ = librosa.load(file_name1, sr=sr)
    audio2, _ = librosa.load(file_name2, sr=sr)

    max_len = max(len(audio1), len(audio2))
    audio1 = librosa.util.fix_length(audio1, size=max_len)
    audio2 = librosa.util.fix_length(audio2, size=max_len)

    mixed_audio = audio1 + audio2
    sf.write(out_path, mixed_audio, sr)


def spotify_search(src, tgt, output_file_name, client_id, client_secret):
    # request API access token
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
        access_token = token_data["access_token"]
        print("Access Token:", access_token)
    else:
        print("Error:", response.status_code)

    # POST query
    query = ["remaster"]
    for key in src:
        if key in ["track", "album", "artist", "genre"]:
            value = " ".join(src[key])
            query.append(f"{key}:{value}")

    if tgt == "playlist":
        query[0] = src["description"][0]

    query = " ".join(query).replace(" ", "%20")
    query = urllib.parse.quote(query)
    url = f"https://api.spotify.com/v1/search?query={query}&type={tgt}"
    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()[tgt + "s"]["items"][0]
        text = dict()
        spotify_id = data["id"]
        text[tgt] = [data["name"]]
        if tgt == "track":
            url = data["preview_url"]
            with open(output_file_name, "wb") as f:
                f.write(requests.get(url).content)

            text["album"] = [data["album"]["name"]]
            text["artist"] = [d["name"] for d in data["artists"]]
        
        if tgt == "album":
            text["date"] = [data["release_date"]]
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
            
        return text
    else:
        print('Response Failed: ', response.status_code)
        return None


def lyric_format(text):
    text = text.split('\n\n')
    delimiters = "\n|,.;?!，。；、？！"
    text = [re.split("["+delimiters+"]", chap) for chap in text]
    
    i = 0
    while i < len(text):
        if len(text[i]) == 1:
            text.pop(i)
            continue
        if len(text[i]) > 4:
            text[i] = text[i][1:]
        i += 1

    return ' '.join([' '.join(chap) for chap in text]).split()
