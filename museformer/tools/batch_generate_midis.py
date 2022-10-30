import os
import argparse
import midiprocessor as mp


class MidiGenerator(object):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--encoding-method', required=True)
        parser.add_argument('--process-token-str-list-method', default='default')
        parser.add_argument('--input-dir', required=True)
        parser.add_argument('--output-dir')
        parser.add_argument('--suffix', default='.txt')
        parser.add_argument('--skip-error', type=lambda x: x.lower() == 'true', default=True)

    @classmethod
    def build(cls, args):
        return cls(args.encoding_method, process_token_str_list_method=args.process_token_str_list_method)

    def __init__(self, encoding_method, process_token_str_list_method='default'):
        self.encoding_method = encoding_method
        self.process_token_str_list_method = process_token_str_list_method
        self.process_token_str_list_func = self.get_process_token_str_list_func(self.process_token_str_list_method)
        self.midi_decoder = mp.MidiDecoder(self.encoding_method)

    def do(self, token_text_path, output_midi_path):
        with open(token_text_path, 'r') as f:
            x = f.read()
        line = self._process_token_str(x)
        midi_obj = self.midi_decoder.decode_from_token_str_list(line)
        dir_name = os.path.dirname(output_midi_path)
        if dir_name not in ('', '.'):
            os.makedirs(dir_name, exist_ok=True)
        midi_obj.dump(output_midi_path)

    def do_batch(self, token_output_dir, midi_output_dir, suffix='.txt', skip_error=True):
        count = 0
        error_count = 0
        for root_dir, dirs, files in os.walk(token_output_dir):
            for file_name in files:
                if not file_name.endswith(suffix):
                    continue
                file_path = os.path.join(root_dir, file_name)
                relative_file_path = os.path.relpath(file_path, token_output_dir)

                base_name = file_name
                save_dir = os.path.dirname(os.path.join(midi_output_dir, relative_file_path))
                save_path = os.path.join(save_dir, base_name + '.mid')

                print('parsing', file_path)
                # noinspection PyBroadException
                try:
                    self.do(file_path, save_path)
                except KeyboardInterrupt:
                    raise
                except:
                    print('Error:', file_path)
                    error_count += 1
                    import traceback
                    traceback.print_exc()
                    if skip_error:
                        continue
                else:
                    count += 1

        return count, error_count

    def _process_token_str(self, x):
        x = x.split('\n')[0]
        x = x.strip().split(' ')
        x = self.process_token_str_list_func(x)
        return x

    @classmethod
    def get_process_token_str_list_func(cls, method):
        if method == 'default':
            return cls.default_process_token_str_list
        else:
            raise ValueError(method)

    @staticmethod
    def default_process_token_str_list(x):
        return x


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    MidiGenerator.add_args(parser)

    args = parser.parse_args()

    midi_generator = MidiGenerator.build(args)
    count, error_count = midi_generator.do_batch(
        args.input_dir, getattr(args, 'output_dir', args.input_dir),
        suffix=args.suffix, skip_error=args.skip_error
    )
    print('Done. %d succeed! %d failed.' % (count, error_count))


if __name__ == '__main__':
    main()
