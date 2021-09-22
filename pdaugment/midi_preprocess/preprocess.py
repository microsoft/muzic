import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings('ignore')

from midi_preprocess.steps.process_midi_file import process_midi_file
from midi_preprocess.steps.filter_and_merge import filter_and_merge
from midi_preprocess.steps.merge_tracks_for_train import gen_merge_midi

from midi_preprocess.utils.hparams import hparams, set_hparams


if __name__ == "__main__":
    set_hparams()
    raw_data_dir = hparams['raw_data_dir']
    processed_data_dir = hparams['processed_data_dir']
    track_info = hparams['instru2program']
    step_per_bar = hparams['step_per_bar']
    instru2track = hparams['instru2track']

    process_midi_file(raw_data_dir, processed_data_dir, hparams)
    filter_and_merge(processed_data_dir, track_info)
    gen_merge_midi(processed_data_dir, step_per_bar, track_info, instru2track)