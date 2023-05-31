import os

from . import const
from .vocab_manager import VocabManager
from . import data_utils

ENCODINGS = ('REMIGEN', 'REMIGEN2')


def raise_encoding_method_error(encoding_method):
    raise ValueError("Encoding method %s is not supported." % encoding_method)


def check_encoding_method(encoding_method):
    assert encoding_method in ENCODINGS, \
        "Encoding method %s not in the supported: %s" % (encoding_method, ', '.join(ENCODINGS))


class MidiDecoder:
    def __init__(
        self,
        encoding_method,
    ):
        check_encoding_method(encoding_method)
        self.encoding_method = encoding_method

        self.vm = VocabManager()

    # === Vocab Properties ===
    @property
    def pos_resolution(self):
        return self.vm.pos_resolution

    @property
    def max_ts_denominator(self):
        return self.vm.max_ts_denominator

    @property
    def max_notes_per_bar(self):
        return self.vm.max_notes_per_bar

    @property
    def beat_note_factor(self):
        return self.vm.beat_note_factor

    # Finished
    # encoding_method check
    def convert_token_str_list_to_token_list(self, token_str_list):
        """

        :param token_str_list:
        :return:
        """
        if self.encoding_method == 'REMI':
            # from . import enc_remi_utils
            # return enc_remi_utils.convert_remi_token_str_list_to_token_list(token_str_list)
            raise NotImplementedError
        elif self.encoding_method == 'REMIGEN':
            from . import enc_remigen_utils
            return enc_remigen_utils.convert_token_str_list_to_token_list(token_str_list)
        elif self.encoding_method == 'REMIGEN2':
            from . import enc_remigen2_utils
            return enc_remigen2_utils.convert_token_str_list_to_token_list(token_str_list)
        elif self.encoding_method == 'CP2':
            # from . import enc_cp2_utils
            # return enc_cp2_utils.convert_token_str_list_to_token_list(token_str_list)
            raise NotImplementedError
        else:
            raise ValueError("convert_token_str_list_to_token_list method is not implemented for encoding method %s"
                             % self.encoding_method)

    def decode_file(
        self,
        file_path,
        ticks_per_beat=const.DEFAULT_TICKS_PER_BEAT,
        ts=const.DEFAULT_TS,
        tempo=const.DEFAULT_TEMPO,
        inst_id=const.DEFAULT_INST_ID,
        velocity=const.DEFAULT_VELOCITY,
        save_path=None,
        **kwargs,
    ):
        """
        Decode MIDI from token str file for one song.
        :param file_path: The file path of the file that contains encodings for one song. Each line contains tokens of
                          one piece.
        :param combine_pieces: Whether to combine those pieces into one song.
        :param ticks_per_beat:
        :param ts:
        :param tempo:
        :param inst_id:
        :param velocity:
        :param save_path: Place to save MIDI file.
        :param scheme_name:
        :return: List of MIDI obj, or MIDI obj if combine_pieces.
        """

        token_str_lists = data_utils.load_lists(file_path, keep_full_dim=False)
        midi_results = self.decode_from_token_str_lists(
            token_str_lists,
            ticks_per_beat=ticks_per_beat,
            ts=ts,
            tempo=tempo,
            inst_id=inst_id,
            velocity=velocity,
            **kwargs,
        )

        if save_path is not None:
            data_utils.ensure_file_dir_to_save(save_path)
            if len(midi_results) == 1:
                midi_results[0].dump(save_path)
            else:
                basename = os.path.basename(save_path)
                basename_split = basename.split('.')
                if len(basename_split) == 0:
                    file_path = save_path + '.%d.mid'
                else:
                    file_path = os.path.join(os.path.dirname(save_path),
                                             '.'.join(basename_split[:-1]) + '.%d.' + basename_split[-1])
                for idx, midi_result in enumerate(midi_results):
                    midi_result.dump(file_path % idx)

        return midi_results

    def decode_from_token_str_lists(
        self,
        token_str_lists,
        ticks_per_beat=const.DEFAULT_TICKS_PER_BEAT,
        ts=const.DEFAULT_TS,
        tempo=const.DEFAULT_TEMPO,
        inst_id=const.DEFAULT_INST_ID,
        velocity=const.DEFAULT_VELOCITY,
        **kwargs,
    ):
        """
        Decode MIDI from token str lists for one file.
        :param token_str_lists: The token_str_lists of the tokens for one song.
        :param combine_pieces: Whether to combine those pieces into one song.
        :param ticks_per_beat:
        :param ts:
        :param tempo:
        :param inst_id:
        :param velocity:
        :return: List of MIDI obj, or MIDI obj if combine_pieces.
        """

        num_layer = data_utils.check_list_layers(token_str_lists, valid_iterable=(list,))
        assert 1 <= num_layer < 3, "Only support decoding for one song. " \
                                   "The token lists may contain tokens for more than one song."
        if num_layer == 1:  # for a song containing only one piece.
            token_str_lists = [token_str_lists]

        for idx, token_list in enumerate(token_str_lists):
            assert len(token_list) > 0, "Piece %d in the token sequences is empty." % idx

        midi_objs = []
        for idx, token_str_list in enumerate(token_str_lists):
            midi_obj = self.decode_from_token_str_list(
                token_str_list,
                ticks_per_beat=ticks_per_beat,
                ts=ts,
                tempo=tempo,
                inst_id=inst_id,
                velocity=velocity,
                **kwargs,
            )
            midi_objs.append(midi_obj)
        return midi_objs

    # Finished
    def decode_from_token_str_list(
        self,
        token_str_list,
        ticks_per_beat=const.DEFAULT_TICKS_PER_BEAT,
        ts=const.DEFAULT_TS,
        tempo=const.DEFAULT_TEMPO,
        inst_id=const.DEFAULT_INST_ID,
        velocity=const.DEFAULT_VELOCITY,
        **kwargs,
    ):
        """
        Decode MIDI obj from token_str_list.
        :param token_str_list: The token_str_list of the tokens for one piece.
        :param ticks_per_beat:
        :param ts:
        :param tempo:
        :param inst_id:
        :param velocity:
        :param scheme_name:
        :return: Midi obj.
        """

        token_list = self.convert_token_str_list_to_token_list(token_str_list)
        midi_obj = self.decode_from_token_list(
            token_list,
            ticks_per_beat=ticks_per_beat,
            ts=ts,
            tempo=tempo,
            inst_id=inst_id,
            velocity=velocity,
            **kwargs,
        )

        return midi_obj

    def decode_from_token_list(
        self,
        token_list,
        ticks_per_beat=const.DEFAULT_TICKS_PER_BEAT,
        ts=const.DEFAULT_TS,
        tempo=const.DEFAULT_TEMPO,
        inst_id=const.DEFAULT_INST_ID,
        velocity=const.DEFAULT_VELOCITY,
        **kwargs,
    ):
        """

        :param token_list:
        :param ticks_per_beat:
        :param ts:
        :param tempo:
        :param inst_id:
        :param velocity:
        :param scheme_name:
        :return:
        """

        if self.encoding_method == 'REMI':
            raise NotImplementedError
            # from . import enc_remi_utils
            # token_list = enc_remi_utils.fix_remi_token_list(token_list)
            # return enc_remi_utils.generate_midi_obj_from_remi_token_list(
            #     token_list, self.vm,
            #     ticks_per_beat=ticks_per_beat,
            #     ts=ts,
            #     tempo=tempo,
            #     inst_id=inst_id,
            #     velocity=velocity,
            # )
        elif self.encoding_method == 'REMIGEN':
            from . import enc_remigen_utils
            token_list = enc_remigen_utils.fix_token_list(token_list)
            return enc_remigen_utils.generate_midi_obj_from_remigen_token_list(
                token_list, self.vm,
                ticks_per_beat=ticks_per_beat,
                ts=ts,
                tempo=tempo,
                inst_id=inst_id,
                velocity=velocity,
                **kwargs,
            )
        elif self.encoding_method == 'REMIGEN2':
            from . import enc_remigen2_utils
            token_list = enc_remigen2_utils.fix_token_list(token_list)
            return enc_remigen2_utils.generate_midi_obj_from_remigen_token_list(
                token_list, self.vm,
                ticks_per_beat=ticks_per_beat,
                ts=ts,
                tempo=tempo,
                inst_id=inst_id,
                velocity=velocity,
                **kwargs,
            )
        elif self.encoding_method == 'CP2':
            raise NotImplementedError
            # from . import enc_cp2_utils
            # return enc_cp2_utils.generate_midi_obj_from_remigen_token_list(
            #     token_list, self.vm,
            # )
        else:
            raise ValueError("decode_from_token_list method is not implemented for encoding method %s"
                             % self.encoding_method)
