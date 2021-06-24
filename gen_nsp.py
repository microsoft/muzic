import os
import sys
import pickle
import preprocess
import random
# (Measure, Pos, Program, Pitch, Duration, Velocity, TimgSig, Tempo)
samples_size = int(input('samples_size = '))  # set to 0 for full generation
velocity_list = [8, 20, 31, 42, 53, 64, 80, 96, 112, 127]
timesig = preprocess.t2e((4, 4))
tempo = preprocess.b2e(120.0)
source_pos_resolution = 32
pos_scale = source_pos_resolution // preprocess.pos_resolution
chord = pickle.load(open('chord_mapping.pkl', 'rb'))
task = input('task = ')
assert task in ['next', 'acc']
input_file_dir = 'PiRhDy/dataset/context_{}'.format(task)
output_file_dir = '{}_data_raw'.format(task)
if os.path.exists(output_file_dir):
    print('Output path {} already exists!'.format(output_file_dir))
    sys.exit(0)
os.system('mkdir -p {}'.format(output_file_dir))
preprocess.gen_dictionary('{}/dict.txt'.format(output_file_dir))
cnt = 0
samples = []
if samples_size > 0:
    with open(input_file_dir + '/' + 'train') as f_input:
        num_lines = sum(1 for line in f_input if len(line.strip()) > 0)
        samples = random.sample(range(num_lines // 2), k=(samples_size * 2))
for split in ['train', 'test']:
    samples_split = set(samples[:samples_size] if split ==
                        'train' else samples[samples_size:])
    with open(input_file_dir + '/' + ('train' if samples_size > 0 else split)) as f_input:
        with open(output_file_dir + '/' + split + '.txt', 'w') as f_txt:
            with open(output_file_dir + '/' + split + '.label', 'w') as f_label:
                idx = 0
                for file_line in f_input:
                    if samples_size > 0:
                        if (idx // 2) not in samples_split:
                            idx += 1
                            continue
                        else:
                            idx += 1
                    else:
                        idx += 1
                    data_item = list(map(int, file_line.strip().split(',')))
                    first_length = sum(
                        _ > 0 for _ in data_item[:source_pos_resolution * 4])
                    encoding = []
                    for i in range(2):
                        for j in range(source_pos_resolution * 4):
                            l = [
                                data_item[((i * 4 + _) * source_pos_resolution * 4) + j] for _ in range(4)]
                            if ((j == 0 and l[3] == 2) or l[3] == 3) and l[0] in chord:
                                pos = (j if task == 'acc' or i ==
                                       0 else first_length + j) * pos_scale
                                measure = pos // (preprocess.pos_resolution *
                                                  preprocess.beat_note_factor)
                                pos = pos % (
                                    preprocess.pos_resolution * preprocess.beat_note_factor)
                                velocity = preprocess.v2e(
                                    velocity_list[l[2] - 1])
                                k = j + 1
                                while k < source_pos_resolution * 4 and data_item[((i * 4 + 3) * source_pos_resolution * 4) + k] in [1, 2]:
                                    k += 1
                                duration = preprocess.d2e((k - j) * pos_scale)
                                s = chord[l[0]] if type(
                                    chord[l[0]]) == set else set(range(12))
                                program = 0 if task == 'acc' and i > 0 else 80
                                for c in s:
                                    pitch = c + ((l[1] - 1) * 12)
                                    encoding.append(
                                        (measure, pos, program, pitch, duration, velocity, timesig, tempo))
                    encoding = list(sorted(encoding))
                    print(preprocess.encoding_to_str(encoding), file=f_txt)
                    print(data_item[-1], file=f_label)  # 1=True 0=False
                    cnt += 1
                    if cnt % 1000 == 0:
                        print(split, cnt, idx)
