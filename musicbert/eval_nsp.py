# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from fairseq.models.roberta import RobertaModel
import numpy as np
import torch
import torch.nn.functional as F
import sys

batch_size = 16

n_samples = None

print('loading model and data')

roberta = RobertaModel.from_pretrained(
    '.',
    checkpoint_file=sys.argv[1],
    data_name_or_path=sys.argv[2],
    user_dir='musicbert'
)
num_classes = 2
group_size = 50
roberta.task.load_dataset('valid')
dataset = roberta.task.datasets['valid']
label_dict = roberta.task.label_dictionary
pad_index = label_dict.pad()


def label_fn(label): return label_dict.string(
    [label + label_dict.nspecial]
)


roberta.cuda()
roberta.eval()

cnt = 0

y_true = []
y_pred = []


def padded(seq, max_length):
    pad_length = max_length - seq.shape[0]
    assert pad_length >= 0
    return np.concatenate((seq, np.full((pad_length,), pad_index, dtype=seq.dtype)))


assert len(dataset) % group_size == 0

for i in range(0, len(dataset), batch_size):
    if n_samples and i == group_size * n_samples:
        break
    target = np.vstack(tuple(dataset[j]['target'].numpy()
                             for j in range(i, i + batch_size) if j < len(dataset)))
    target = torch.from_numpy(target)
    target = F.one_hot(target.long(), num_classes=num_classes)
    target = target.sum(dim=1)
    source_batch_max_length = max(dataset[j]['net_input.src_tokens'].size(
        0) for j in range(i, i + batch_size) if j < len(dataset))
    source = np.vstack(tuple(padded(dataset[j]['net_input.src_tokens'].numpy(
    ), source_batch_max_length) for j in range(i, i + batch_size) if j < len(dataset)))
    source = torch.from_numpy(source)
    output = F.softmax(roberta.predict(
        'acc_head' if 'acc' in sys.argv[1] else 'next_head', source, True), dim=1)
    y_true.append(target.detach().cpu().numpy())
    y_pred.append(output.detach().cpu().numpy())
    print('evaluating: {:.4f}%'.format(
        i / len(dataset) * 100), end='\r', flush=True)

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

print()

for i in range(num_classes):
    print('label[{}] ='.format(i), label_fn(i))

print('y_true.shape =', y_true.shape)
print('y_pred.shape =', y_pred.shape)

query_list = []

assert y_pred.shape[0] % group_size == 0

assert {label_fn(0), label_fn(1)} == {'0', '1'}

for i in range(0, y_pred.shape[0], group_size):
    x = 1 if int(label_fn(1)) == 1 else 0  # find which label is "true"
    q = tuple((-y_pred[j][x], y_true[j][x]) for j in range(i, i + group_size))
    q = tuple(j[1] for j in sorted(q))
    query_list.append(q)


def AP(q):
    rk_list = []
    for i, j in enumerate(q):
        if j == 1:
            rk_list.append(i + 1)
    # print(rk_list)
    result = sum((i + 1) / j for i, j in enumerate(rk_list)) / len(rk_list)
    return result


print('MAP: {:.6f}'.format(sum(AP(q) for q in query_list) / len(query_list)))

for z in [1, 5, 10, 15, 20, 25]:
    print('HITS@{}: {:.6f}'.format(z,
                                   sum(sum(q[:z]) / sum(q) for q in query_list) / len(query_list)))

with open(sys.argv[1].split('/')[-1].split('.')[0] + '.npy', 'wb') as f:
    np.save(f, {'y_true': y_true, 'y_pred': y_pred})
