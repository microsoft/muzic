import math
import numpy as np

from .utils.magenta_chord_recognition import infer_chords_for_sequence, _key_chord_distribution, \
    _key_chord_transition_distribution


class Item(object):
    def __init__(self, name, start, end, vel=0, pitch=0, track=0, value=''):
        self.name = name
        self.start = start  # start step
        self.end = end  # end step
        self.vel = vel
        self.pitch = pitch
        self.track = track
        self.value = value

    def __repr__(self):
        return f'Item(name={self.name:>10s}, start={self.start:>4d}, end={self.end:>4d}, ' \
               f'vel={self.vel:>3d}, pitch={self.pitch:>3d}, track={self.track:>2d}, ' \
               f'value={self.value:>10s})\n'

    def __eq__(self, other):
        return self.name == other.name and self.start == other.start and \
               self.pitch == other.pitch and self.track == other.track


class ChordDetector(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.pos_resolution = self.encoder.vm.pos_resolution
        self.key_chord_loglik, self.key_chord_transition_loglik = self.init_for_chord_detection()

    @staticmethod
    def init_for_chord_detection():
        chord_pitch_out_of_key_prob = 0.01
        key_change_prob = 0.001
        chord_change_prob = 0.5
        key_chord_distribution = _key_chord_distribution(
            chord_pitch_out_of_key_prob=chord_pitch_out_of_key_prob)
        key_chord_loglik = np.log(key_chord_distribution)
        key_chord_transition_distribution = _key_chord_transition_distribution(
            key_chord_distribution,
            key_change_prob=key_change_prob,
            chord_change_prob=chord_change_prob)
        key_chord_transition_loglik = np.log(key_chord_transition_distribution)
        return key_chord_loglik, key_chord_transition_loglik

    def infer_chord_for_pos_info(self, pos_info):
        # 此函数只针对4/4的曲子
        # input: pos_info, 经过normalize到大小调、去除melody轨道重叠音符并且decode(encode(pos_info))之后的pos_info
        # output: 这个pos_info的和弦，粒度是24个位置一个和弦，一个bar有两个和弦
        # magenta算法已经修复了多一个和弦的bug
        key_chord_loglik, key_chord_transition_loglik = self.key_chord_loglik, self.key_chord_transition_loglik
        pos_resolution = self.pos_resolution

        max_pos = 0
        note_items = []
        for bar, ts, pos, tempo, insts_notes in pos_info:
            if ts is not None and tuple(ts) != (4, 4):
                raise NotImplementedError("This implementation only supports time signature 4/4.")
            if insts_notes is None:
                continue
            for inst_id in insts_notes:
                if inst_id >= 128:
                    continue
                inst_notes = insts_notes[inst_id]  # 浅复制，修改会影响值
                for note_idx, (pitch, duration, velocity) in enumerate(inst_notes):
                    max_pos = max(max_pos, bar * pos_resolution * 4 + pos + duration)
                    if 0 <= pitch < 128:
                        # squeeze pitch ranges to facilitate chord detection
                        while pitch > 72:
                            pitch -= 12
                        while pitch < 48:
                            pitch += 12
                        note_items.append(
                            Item(
                                name='On',
                                start=bar * pos_resolution * 4 + pos,  # 这里pos_resolution*4代表一个bar有4拍，不是4/4的曲子不适用
                                end=bar * pos_resolution * 4 + pos + duration,
                                vel=velocity,
                                pitch=pitch,
                                track=0
                            )
                        )
        note_items.sort(key=lambda x: (x.start, -x.end))
        pos_per_chord = pos_resolution * 2  # 24
        # max_chords = round(max_pos // pos_per_chord + 0.5)
        max_chords = math.ceil(max_pos / pos_per_chord)
        chords = infer_chords_for_sequence(
            note_items,
            pos_per_chord=pos_per_chord,
            max_chords=max_chords,
            key_chord_loglik=key_chord_loglik,
            key_chord_transition_loglik=key_chord_transition_loglik
        )
        num_bars = pos_info[-1][0] + 1
        while len(chords) < num_bars * 2:
            chords.append('N.C.')
        if len(chords) > num_bars * 2:  # with a very long note in th end, the chords num will be larger than num_bars*2
            chords = chords[:num_bars * 2]
        assert len(chords) == num_bars * 2, 'chord length: %d, number of bars: %d' % (len(chords), num_bars)
        return chords
