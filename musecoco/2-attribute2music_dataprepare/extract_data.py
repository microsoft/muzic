import argparse
import os
from tqdm.auto import tqdm
import msgpack
import json

from file_list import generate_file_list
from config import attribute_list

import midi_data_extractor as mde
# import midiprocessor as mp


def get_midi_file_list(data_dir, suffixes=('.mid', '.midi')):
    return generate_file_list(data_dir, suffixes=suffixes, ignore_suffix_case=True, save_path=None)


def main(data_dir, save_dir):
    file_list = get_midi_file_list(data_dir)
    print('Altogether %d MIDI files...' % len(file_list))
    extractor = mde.DataExtractor(None, encoding_method='REMIGEN2', attribute_list=attribute_list)

    os.makedirs(save_dir, exist_ok=True)

    packer_1 = msgpack.Packer()
    packer_2 = msgpack.Packer()

    with open(os.path.join(save_dir, 'RID.bin'), 'wb') as f1, open(os.path.join(save_dir, 'TOKEN.bin'), 'wb') as f2, open(os.path.join(save_dir, 'file_list.txt'), 'w', encoding='utf-8') as f3:
        n_success = 0
        for file_path in tqdm(file_list):
            try:
                tokens, _, _, info_dict, _ = extractor.extract(
                    data_dir, file_path,
                    cut_method='random_2',
                    normalize_pitch_value=True,

                    # === load values for the subjective attributes here ===
                    artist=None,  # 'mozart',
                    genre=None,  # ('Pop_Rock', 'RnB'), 
                    emotion=None,  # 'Q1',
                    # =============
                )
            except:
                import traceback
                traceback.print_exc()
                # raise
            else:
                n_success += 1
                f1.write(packer_1.pack(info_dict))
                f2.write(packer_2.pack(tokens))
                f3.write(file_path + '\n')
            
    def write_index(bin_path, save_path):
        index = []
        with open(bin_path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, use_list=False)
            index.append((unpacker.tell()))
            for idx, sample in enumerate(unpacker):
                index.append((unpacker.tell()))
        index = index[:-1]
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(index, f)
    
    write_index(os.path.join(save_dir, 'RID.bin'), os.path.join(save_dir, 'RID_index.json'))
    write_index(os.path.join(save_dir, 'TOKEN.bin'), os.path.join(save_dir, 'TOKEN_index.json'))
    
    print('Done. %d files succeeded. Stored in %s' % (n_success, save_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('save_dir')
    args = parser.parse_args()
    main(args.data_dir, args.save_dir)
