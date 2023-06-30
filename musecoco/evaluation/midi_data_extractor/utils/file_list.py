import os


def dump_file_list(file_list, save_path):
    dirname = os.path.dirname(save_path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    with open(save_path, 'w') as f:
        for item in file_list:
            f.write(item + '\n')


def generate_file_list(dir, suffixes=None, ignore_suffix_case=True, save_path=None):
    file_list = []
    for root_dir, _, files in os.walk(dir):
        for file_name in files:
            if suffixes is not None:
                skip = True
                for sf in suffixes:
                    if ignore_suffix_case:
                        sf = sf.lower()
                        fn = file_name.lower()
                    else:
                        fn = file_name
                    if fn.endswith(sf):
                        skip = False
                        break
                if skip:
                    continue
            
            file_path = os.path.join(root_dir, file_name).replace('\\', '/')
            file_path = os.path.relpath(file_path, dir)
            file_list.append(file_path)
    if save_path is not None:
        dump_file_list(file_list, save_path)
    return file_list


def read_file_list(path):
    file_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            file_list.append(line)
    return file_list
