import json, pickle
from copy import deepcopy

test = json.load(open('data/predict.json','r'))
pred = json.load(open('predict_attributes.json','r'))
probs = json.load(open('softmax_probs.json','r'))
att_key = json.load(open('data/att_key.json','r'))

final = []
for line in test:
    ins = {}
    ins['text'] = line['text']
    ins['pred_labels'] = {}
    ins['pred_probs'] = {}
    final.append(deepcopy(ins))

for k, v in pred.items():
    for j in range(len(v)):
        final[j]['pred_labels'][k] = deepcopy(v[j])
for k, v in probs.items():
    for j in range(len(v)):
        final[j]['pred_probs'][k] = deepcopy(v[j])

I1s2_key = []
S4_key = []
for att in att_key:
    if att[:4]=="I1s2":
        I1s2_key.append(att)
    if att[:2]=="S4":
        S4_key.append(att)

for idx in range(len(final)):
    pred_labels_I1s2 = []
    pred_probs_I1s2 = []
    pred_labels_S4 = []
    pred_probs_S4 = []
    for i1s2 in I1s2_key:
        pred_labels_I1s2.append(deepcopy(final[idx]['pred_labels'][i1s2]))
        pred_probs_I1s2.append(deepcopy(final[idx]['pred_probs'][i1s2]))
        final[idx]['pred_labels'].pop(i1s2)
        final[idx]['pred_probs'].pop(i1s2)
    for s4 in S4_key:
        pred_labels_S4.append(deepcopy(final[idx]['pred_labels'][s4]))
        pred_probs_S4.append(deepcopy(final[idx]['pred_probs'][s4]))
        final[idx]['pred_labels'].pop(s4)
        final[idx]['pred_probs'].pop(s4)
    final[idx]['pred_probs']['I1s2'] = deepcopy(pred_probs_I1s2)
    final[idx]['pred_probs']['S4'] = deepcopy(pred_probs_S4)
    final[idx]['pred_labels']['I1s2'] = deepcopy(pred_labels_I1s2)
    final[idx]['pred_labels']['S4'] = deepcopy(pred_labels_S4)

pickle.dump(final, open('stage1.bin','wb'))