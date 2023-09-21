from midiprocessor import midi_utils, MidiEncoder, enc_remigen_utils, enc_remigen2_utils


def get_midi_pos_info(
    encoder,
    midi_path=None,
    midi_obj=None,
    remove_empty_bars=True,
):
    if midi_obj is None:
        midi_obj = midi_utils.load_midi(midi_path)

    pos_info = encoder.collect_pos_info(midi_obj, trunc_pos=None, tracks=None, remove_same_notes=False, end_offset=0)
    del midi_obj

    # encode and decode to ensure the info consistency, i.e., let the following chord and cadence detection
    # happen on exactly the same info as the resulting token sequences
    pos_info = encoder.convert_pos_info_to_pos_info_id(pos_info)
    pos_info = encoder.convert_pos_info_id_to_pos_info(pos_info)
    # remove the beginning and ending empty bars
    if remove_empty_bars:
        pos_info = encoder.remove_empty_bars_for_pos_info(pos_info)

    return pos_info


def convert_pos_info_to_tokens(encoder, pos_info, **kwargs):
    pos_info_id = encoder.convert_pos_info_to_pos_info_id(pos_info)
    if encoder.encoding_method == 'REMIGEN':
        enc_utils = enc_remigen_utils
    elif encoder.encoding_method == 'REMIGEN2':
        enc_utils = enc_remigen2_utils
    else:
        raise ValueError(encoder.encoding_method)
    tokens = enc_utils.convert_pos_info_to_token_lists(
        pos_info_id, ignore_ts=False, sort_insts='id', sort_notes=None, **kwargs
    )[0]
    tokens = enc_utils.convert_remigen_token_list_to_token_str_list(tokens)
    return tokens


if __name__ == '__main__':
    midi_path = 'test.mid'

    enc = MidiEncoder("REMIGEN")
    
    pi = get_midi_pos_info(enc, midi_path)
    # 这是一个包含MIDI全部信息的list，【你所需要的信息理论上从这个里面获取最方便】
    # 列表的长度为该MIDI的最大position个数
    # 例如某个MIDI只有1个bar，这个bar有4拍，程序设定每拍分为12个position，那么pos_info的大小为1*4*12=48
    # pos_info中的每个元素也是一个列表，长度为5，信息依次为：
    #     bar index: bar的索引，从0开始
    #     ts: Time signature，只有在有变化的时候才会有，否则None
    #     local_pos: bar内的onset位置，例如在一个4拍的bar中，该数字会从0一直到47
    #     tempo: Tempo的值，只有在有变化的时候才会有，否则None
    #     insts_notes: 是一个字典，key为inst的id（鼓为128），value为该位置所有音符的集合，信息包含（pitch, duration, velocity）
    # 可以自己弄一个MIDI来试试
    # 温馨提示：如果下一次还需要这个MIDI的信息，可以保存pos_info，下次直接加载就能更快一些

    # 转化为token序列
    tokens = convert_pos_info_to_tokens(enc, pi)

    print(tokens[:100])
