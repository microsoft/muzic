from fairseq.models.roberta import RobertaModel
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics
import sys
import os

max_length = 8192 if 'disable_cp' not in os.environ else 1024
batch_size = 4
n_folds = 1

scores = dict()
for score in ["f1_score", "roc_auc_score"]:
    for average in ["macro", "micro", "weighted", "samples"]:
        scores[score + "_" + average] = []

for i in range(n_folds):

    print('loading model and data')
    print('start evaluating fold {}'.format(i))

    roberta = RobertaModel.from_pretrained(
        '.',
        checkpoint_file=sys.argv[1].replace('x', str(i)),
        data_name_or_path=sys.argv[2].replace('x', str(i)),
        user_dir='musicbert'
    )
    num_classes = 13 if 'topmagd' in sys.argv[1] else 25
    roberta.task.load_dataset('valid')
    dataset = roberta.task.datasets['valid']
    label_dict = roberta.task.label_dictionary
    pad_index = label_dict.pad()
    label_fn = lambda label: label_dict.string(
        [label + label_dict.nspecial]
    )
    roberta.cuda()
    roberta.eval()

    cnt = 0

    y_true = []
    y_pred = []

    def padded(seq):
        pad_length = max_length - seq.shape[0]
        assert pad_length >= 0
        return np.concatenate((seq, np.full((pad_length,), pad_index, dtype=seq.dtype)))

    for i in range(0, len(dataset), batch_size):
        target = np.vstack(tuple(padded(dataset[j]['target'].numpy()) for j in range(i, i + batch_size) if j < len(dataset)))
        target = torch.from_numpy(target)
        target = F.one_hot(target.long(), num_classes=(num_classes + 4))
        target = target.sum(dim=1)[:, 4:]
        source = np.vstack(tuple(padded(dataset[j]['source'].numpy()) for j in range(i, i + batch_size) if j < len(dataset)))
        source = torch.from_numpy(source)
        output = torch.sigmoid(roberta.predict('topmagd_head' if 'topmagd' in sys.argv[1] else 'masd_head', source, True))
        y_true.append(target.detach().cpu().numpy())
        y_pred.append(output.detach().cpu().numpy())
        print('evaluating: {:.2f}%'.format(i / len(dataset) * 100), end='\r', flush=True)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    print()

    for i in range(num_classes):
        print(i, label_fn(i))

    print(y_true.shape)
    print(y_pred.shape)

    # with open('genre.npy', 'wb') as f:
    #    np.save(f, {'y_true': y_true, 'y_pred': y_pred})

    for score in ["f1_score", "roc_auc_score"]:
        for average in ["macro", "micro", "weighted", "samples"]:
            try:
                y_score = np.round(y_pred) if score == "f1_score" else y_pred
                result = sklearn.metrics.__dict__[score](y_true, y_score, average=average)
                print("{}_{}:".format(score, average), result)
                scores[score + "_" + average].append(result)
            except BaseException as e:
                print("{}_{}:".format(score, average), e)
                scores[score + "_" + average].append(None)


print(scores)
for k in scores:
    print(k, sum(scores[k]) / len(scores[k]))

