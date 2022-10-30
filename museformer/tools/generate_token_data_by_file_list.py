import os
import argparse
from tqdm import tqdm


def read_file_list(file_list):
    path_list = []
    with open(file_list, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            path_list.append(line)
    return path_list


def main(file_list, token_dir, save_dir, file_list_suffix='.txt'):
    file_list_base = os.path.basename(file_list)[:-len(file_list_suffix)]
    file_list = read_file_list(file_list)
    # print('processing %d files...' % len(file_list))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, file_list_base + '.data'), 'w') as save_f:
        for file_path in tqdm(file_list):
            file_name = os.path.basename(file_path)
            token_path = os.path.join(token_dir, file_name + '.txt')
            with open(token_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue
                    save_f.write(line + '\n')
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_list')
    parser.add_argument('token_dir')
    parser.add_argument('save_dir')
    parser.add_argument('--file_list_suffix', default='.txt')

    args = parser.parse_args()

    main(args.file_list, args.token_dir, args.save_dir, args.file_list_suffix)

    print('Done')
