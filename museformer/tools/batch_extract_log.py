import os
import argparse
import re


def load_midi_processor():
    if midi_processor_cls is not None:
        return midi_processor_cls

    import midiprocessor
    midi_processor_cls = midiprocessor
    return midi_processor_cls


class GenerationLogExtractor(object):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--base_name', default='random')
        parser.add_argument('--start_idx', type=int, default=1)
        parser.add_argument('--process-token-str-method', default='default')

    @classmethod
    def build(cls, args):
        return cls(base_name=args.base_name, start_idx=args.start_idx,
                   process_token_str_method=args.process_token_str_method)

    def __init__(self, base_name='random', start_idx=1, process_token_str_method='default'):
        self.base_name = base_name
        self.start_idx = start_idx
        self.process_token_str_method = process_token_str_method

    def do(self, log_path, token_output_dir, base_name=None, start_idx=None, process_token_str_method=None):
        if base_name is None:
            base_name = self.base_name
        if start_idx is None:
            start_idx = self.start_idx
        if process_token_str_method is None:
            process_token_str_method = self.process_token_str_method
        process_token_str_func = self.get_process_token_str_func(process_token_str_method)
        return self.extract_midi_tokens_from_output_log(
            log_path, token_output_dir, base_name, start_idx, process_token_str_func
        )

    @classmethod
    def get_process_token_str_func(cls, method):
        if method == 'default':
            return cls.default_process_token_str
        else:
            raise ValueError(method)

    @staticmethod
    def default_process_token_str(token_str):
        return token_str.strip()

    @staticmethod
    def extract_midi_tokens_from_output_log(log_path, token_output_dir, base_name, start_idx, process_token_str):
        with open(log_path, 'r') as f:
            s = f.read()
        r = re.findall('D-\d+?\t.+?\t(.+?)\n', s)

        os.makedirs(token_output_dir, exist_ok=True)
        for idx, token_str in enumerate(r, start=start_idx):
            token_str = process_token_str(token_str)
            with open(os.path.join(token_output_dir, '%s-%d.txt') % (base_name, idx), 'w') as f:
                f.write(token_str)
        num_songs = len(r)
        print('Extract %d songs from log. (%s-%d ~ %s-%d)' %
              (num_songs, base_name, start_idx, base_name, start_idx + num_songs - 1))


def main():
    parser = argparse.ArgumentParser()
    GenerationLogExtractor.add_args(parser)
    parser.add_argument('log_path')
    parser.add_argument('token_output_dir')
    args = parser.parse_args()
    generation_log_extractor = GenerationLogExtractor.build(args)
    generation_log_extractor.do(args.log_path, args.token_output_dir)


if __name__ == '__main__':
    main()
