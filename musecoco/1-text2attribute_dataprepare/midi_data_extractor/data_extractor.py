import os
import pickle
import random
import midiprocessor as mp
from copy import deepcopy

from .midi_processing import get_midi_pos_info, convert_pos_info_to_tokens
from .chord_detection import ChordDetector
from .config import attribute_versions_list
from .verbalizer import Verbalizer
from .attribute_unit import load_unit_class
from .utils.pos_process import fill_pos_ts_and_tempo_


def cut_by_none(num_bars, k, min_bar, max_bar):
    return [(0, num_bars)]


def cut_by_random_1(num_bars, k, min_bar, max_bar, auto_k=True):
    if num_bars < min_bar:
        return None
    r = set()
    for begin in range(num_bars - min_bar + 1):
        for end in range(begin + 1, min(begin + max_bar, num_bars)):
            r.add((begin, end))
    if auto_k:
        k = min(len(r), k)
    r = random.sample(r, k)
    return r


def cut_by_random_2(num_bars, k, min_bar, max_bar, auto_k=True):
    if num_bars < min_bar:
        return None
    r = set()
    if num_bars >= max_bar:
        for begin in range(num_bars - max_bar + 1):
            r.add((begin, begin + max_bar))
    else:
        r.add((0, num_bars))
    if auto_k:
        k = min(len(r), k)
    r = random.sample(r, k)
    return r


cut_methods = {
    'none': cut_by_none,
    'random_1': cut_by_random_1,
    'random_2': cut_by_random_2,
}


def get_bar_positions(pos_info):
    r = {}
    for idx, pos_item in enumerate(pos_info):
        bar_id = pos_item[0]
        if bar_id not in r:
            r[bar_id] = [idx, idx]
        r[bar_id][1] = idx + 1
    nr = []
    for idx in range(len(r)):
        nr.append(r[idx])
    r = nr
    return r


def get_bars_insts(pos_info, bars_positions):
    r = []
    num_bars = len(bars_positions)
    for idx in range(num_bars):
        begin, end = bars_positions[idx]
        cur_insts = set()
        for t_idx in range(begin, end):
            notes = pos_info[t_idx][-1]
            if notes is not None:
                for inst_id in notes:
                    cur_insts.add(inst_id)
        cur_insts = tuple(cur_insts)
        r.append(cur_insts)

    return r


class DataExtractor(object):
    def __init__(self, attribute_list_version, encoding_method='REMIGEN', attribute_list=None):
        if encoding_method not in  ('REMIGEN', 'REMIGEN2'):
            raise NotImplementedError("Other encoding method such as %s is not supported yet." % encoding_method)

        self.encoder = mp.MidiEncoder(encoding_method)
        self.chord_detector = ChordDetector(self.encoder)

        if attribute_list is not None:
            self.attribute_list = tuple(set(attribute_list))
        else:
            self.attribute_list = attribute_versions_list[attribute_list_version]
        self.unit_cls_dict = self.init_units(self.attribute_list)

        self.verbalizer = Verbalizer()

    @staticmethod
    def init_units(attribute_list):
        unit_cls_dict = {}
        for attribute_label in attribute_list:
            unit_cls_dict[attribute_label] = load_unit_class(attribute_label)
        return unit_cls_dict

    def extract(
        self, midi_dir, midi_path,
        cut_method='random_1',
        normalize_pitch_value=True,
        pos_info_path=None,
        ignore_chord=False,
        chord_path=None,
        **kwargs,
    ):
        pos_info = None
        loaded_pos_info = False
        if pos_info_path is not None:
            try:
                with open(pos_info_path, 'rb') as f:
                    pos_info = pickle.load(f)
            except FileNotFoundError:
                pos_info = None
        if pos_info is None:
            midi_obj = mp.midi_utils.load_midi(os.path.join(midi_dir, midi_path))
            pos_info = get_midi_pos_info(self.encoder, midi_path=None, midi_obj=midi_obj)
        else:
            loaded_pos_info = True

        pos_info = fill_pos_ts_and_tempo_(pos_info)

        is_major = None
        if normalize_pitch_value:
            try:
                pos_info, is_major, _ = self.encoder.normalize_pitch(pos_info)
            except KeyboardInterrupt:
                raise
            except:
                is_major = None

        # load chord for the whole midi from file, or detect it from the sequence.
        # currently only support midi using only 4/4 measure. else, bars_chords is None.
        bars_chords = None
        loaded_chords = False
        if not ignore_chord:
            if chord_path is not None:
                try:
                    with open(chord_path, 'rb') as f:
                        bars_chords = pickle.load(f)
                except FileNotFoundError:
                    bars_chords = None
            if bars_chords is None:
                try:
                    bars_chords = self.chord_detector.infer_chord_for_pos_info(pos_info)
                except KeyboardInterrupt:
                    raise
                except:
                    bars_chords = None
            else:
                loaded_chords = True

        bars_positions = get_bar_positions(pos_info)
        bars_instruments = get_bars_insts(pos_info, bars_positions)
        num_bars = len(bars_positions)
        assert num_bars == len(bars_instruments)

        attribute_list = self.attribute_list
        unit_cls_dict = self.unit_cls_dict

        length = min(16, num_bars)
        # if length < 4:
        #     raise ValueError("The number of bars is less than 4.")
        pieces_pos = cut_methods[cut_method](num_bars, 3, length, length)  # Todo: allow settings
        if pieces_pos is None:
            print('pieces_pos is None', num_bars, midi_path)
            # assert False
            raise ValueError("No valid pieces are for this MIDI.")

        # Todo: move to better place
        tokens = convert_pos_info_to_tokens(self.encoder, pos_info)
        assert tokens[-1] == 'b-1'
        last_begin = 0
        last_idx = 0
        bars_token_positions = {}
        for idx, token in enumerate(tokens):
            if token == 'b-1':
                bars_token_positions[last_idx] = (last_begin, idx + 1)
                last_begin = idx + 1
                last_idx = last_idx + 1

        pieces = []
        for bar_begin, bar_end in pieces_pos:
            # skip the piece if no any instrument is played.
            seg_insts = bars_instruments[bar_begin: bar_end]
            has_notes = False
            for item in seg_insts:
                if len(item) > 0:
                    has_notes = True
                    break
            if not has_notes:
                continue

            value_dict = {}
            for attribute_label in attribute_list:
                unit_cls = unit_cls_dict[attribute_label]
                unit = unit_cls.new(
                    self.encoder, midi_dir, midi_path, pos_info, bars_positions, bars_chords, bars_instruments,
                    bar_begin, bar_end,  # Todo
                    is_major=is_major,
                    **kwargs,
                )
                value = unit.value
                value_dict[attribute_label] = value

            piece_sample = {
                'bar_begin': bar_begin,
                'bar_end': bar_end,
                'values': value_dict,
                'token_begin': bars_token_positions[bar_begin][0],
                'token_end': bars_token_positions[bar_end - 1][1],
            }
            pieces.append(piece_sample)

        if len(pieces) == 0:
            # assert False
            raise ValueError("No valid results for all the pieces.")

        info_dict = {
            'midi_dir': midi_dir,
            'midi_path': midi_path,
            'pieces': pieces
        }

        loaded_record = {
            'pos_info': loaded_pos_info,
            'chord': loaded_chords,
        }

        return tokens, pos_info, bars_chords, info_dict, loaded_record

    def represent(self, info_dict, remove_raw=False):
        info_dict = deepcopy(info_dict)
        return self.represent_(info_dict, remove_raw=remove_raw)

    def represent_(self, info_dict, remove_raw=False):
        # midi_dir = info_dict['midi_dir']
        midi_path = info_dict['midi_path']
        pieces = info_dict['pieces']
        for piece in pieces:
            value_dict = piece['values']
            unit_dict = piece['units']
            bar_begin = piece['bar_begin']
            bar_end = piece['bar_end']

            reps = []

            try:
                text_list, used_attributes = self.verbalizer.get_text(value_dict)
            except:
                print(midi_path)
                print(bar_begin, bar_end)
                print(value_dict)
                raise

            if len(text_list) == 0:
                continue

            assert len(text_list) == len(used_attributes)

            for text, u_attributes in zip(text_list, used_attributes):
                vectors = {}
                for attribute_label in self.attribute_list:
                    try:
                        unit = unit_dict[attribute_label]
                        if attribute_label in u_attributes:
                            vector = unit.get_vector(use=True, use_info=u_attributes[attribute_label])
                        else:
                            vector = unit.get_vector(use=False, use_info=None)
                        vector = tuple(vector)
                        vectors[attribute_label] = vector
                    except:
                        print('Error while vectorizing "%s".' % attribute_label)
                        raise

                rep_sample = {
                    'text': text,
                    'vectors': vectors
                }

                reps.append(rep_sample)

            piece['reps'] = reps

            if remove_raw:
                piece.pop('values')
                piece.pop('units')

        return info_dict
