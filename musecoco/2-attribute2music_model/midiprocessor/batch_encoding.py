# Author: Botao Yu


import os
import sys
import argparse
from functools import partial

from tqdm import tqdm
import multiprocessing
# from concurrent.futures import ProcessPoolExecutor

from .midi_encoding import MidiEncoder, ENCODINGS as ENC_ENCODINGS
from . import data_utils
from .inst_sorting import INST_SORTING_METHODS
from .note_sorting import NOTE_SORTING_METHODS


def add_args_for_batch_operation(parser):
    parser.add_argument('midi_dir')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--file-list', type=str, default=None)
    parser.add_argument('--midi-suffices', type=lambda x: x.split(','), default=('.mid', '.midi'))
    parser.add_argument('--output-suffix', type=str, default='.txt')
    parser.add_argument('--output-pos-info-id', action='store_true')
    parser.add_argument('--no-skip-error', action='store_true')
    # parser.add_argument('--dump-dict', action='store_true')
    # parser.add_argument('--fairseq-dict', action='store_true')
    parser.add_argument('--num-workers', type=int, default=multiprocessing.cpu_count())


def add_args_for_encoding(parser):
    parser.add_argument('--encoding-method', choices=ENC_ENCODINGS, required=True)
    parser.add_argument('--normalize-pitch-value', action='store_true')
    parser.add_argument('--remove-empty-bars', action='store_true')
    parser.add_argument('--end-offset', type=int, default=0)
    parser.add_argument('--ignore-ts', action='store_true')
    parser.add_argument('--ignore-inst', action='store_true')
    parser.add_argument('--sort-insts', choices=INST_SORTING_METHODS)
    parser.add_argument('--sort-notes', choices=NOTE_SORTING_METHODS)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    add_args_for_batch_operation(parser)
    add_args_for_encoding(parser)
    args = parser.parse_args()

    # === Process ===
    file_path_list = data_utils.get_file_paths(args.midi_dir, file_list=args.file_list,
                                               suffixes=args.midi_suffices)
    num_files = len(file_path_list)
    print('Processing %d files...' % num_files)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    assert num_workers >= 1

    encoder = MidiEncoder(
        encoding_method=args.encoding_method,
    )
    skip_error = not getattr(args, 'no_skip_error', False)

    num_processed = 0
    num_error = 0

    with multiprocessing.Pool(processes=num_workers) as pool:
        iterator = iter(
            tqdm(
                pool.imap(
                    partial(
                        process_file, encoder, args=args, track_dict=None, skip_error=True, save=True
                    ),
                    file_path_list,
                ),
                total=num_files
            )
        )
        for i in range(num_files):
            encodings = next(iterator)
            num_processed += 1
            if encodings is None:
                num_error += 1
                if not skip_error:
                    sys.exit(1)

    print('Done.')
    print('Altogether %d files. Processed %d files. %d succeeded. %d failed.' % (
        num_files, num_processed, num_processed - num_error, num_error)
    )


def process_file(encoder: MidiEncoder, file_path, args, track_dict, skip_error=True, save=False):
    basename = os.path.basename(file_path)
    no_error = True
    try:
        encodings = encoder.encode_file(
            file_path,
            end_offset=getattr(args, 'end_offset', 0),
            normalize_pitch_value=(args, 'normalize_pitch_value', False),
            tracks=None if track_dict is None else track_dict[basename],
            save_pos_info_id_path=(None if not getattr(args, 'output_pos_info_id', False)
                                   else os.path.join(args.output_dir, 'pos_info_id', basename + '.json')),
            remove_empty_bars=getattr(args, 'remove_empty_bars', False),
            ignore_inst=getattr(args, 'ignore_inst', False),
            ignore_ts=getattr(args, 'ignore_ts', False),
            sort_insts=getattr(args, 'sort_insts', None),
            sort_notes=getattr(args, 'sort_notes', None)
        )
        encodings = encoder.convert_token_lists_to_token_str_lists(encodings)
    except KeyboardInterrupt:
        raise
    except:
        tqdm.write('Error when encoding %s.' % file_path)
        no_error = False
        if skip_error:
            import traceback
            tqdm.write(traceback.format_exc())
            encodings = None
        else:
            raise

    if no_error and save:
        output_path = os.path.join(args.output_dir, basename + args.output_suffix)
        data_utils.dump_lists(encodings, output_path, no_internal_blanks=True)

    return encodings


if __name__ == '__main__':
    main()
