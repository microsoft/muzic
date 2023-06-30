import os, argparse, random, json, pickle
from midi_data_extractor import DataExtractor
from midi_data_extractor.attribute_unit import convert_value_dict_into_unit_dict
import numpy as np
from tqdm.auto import tqdm
from midiprocessor import MidiEncoder
from transformers import set_seed
from tqdm import tqdm
from copy import deepcopy


parser = argparse.ArgumentParser()
parser.add_argument('--root', default='.', required=True, help='data dir')
parser.add_argument('--att_key', default='att_key.json')
parser.add_argument('--output', default='.')

args = parser.parse_args()
set_seed(42)

midi_encoder = MidiEncoder("REMIGEN2")

key_order = json.load(open(args.att_key))
I1s2 = ["I1s2_piano", "I1s2_keyboard", "I1s2_percussion", "I1s2_organ", "I1s2_guitar", "I1s2_bass", "I1s2_violin", "I1s2_viola", "I1s2_cello", "I1s2_harp", "I1s2_strings", "I1s2_voice", "I1s2_trumpet", "I1s2_trombone", "I1s2_tuba", "I1s2_horn", "I1s2_brass", "I1s2_sax", "I1s2_oboe", "I1s2_bassoon", "I1s2_clarinet", "I1s2_piccolo", "I1s2_flute", "I1s2_pipe", "I1s2_synthesizer", "I1s2_ethnic_instruments", "I1s2_sound_effects", "I1s2_drum"]
S4 = ["S4_new_age", "S4_electronic", "S4_rap", "S4_religious", "S4_international", "S4_easy_listening", "S4_avant_garde", "S4_rnb", "S4_latin", "S4_children", "S4_jazz", "S4_classical", "S4_comedy_spoken", "S4_pop_rock", "S4_reggae", "S4_stage", "S4_folk", "S4_blues", "S4_vocal", "S4_holiday", "S4_country", "S4_symphony"]
extractor = DataExtractor(encoding_method='REMIGEN2', attribute_list_version='v3')
os.makedirs(args.output, exist_ok=True)


allmidiinfo = {}
allinfermidivector = {}
root = args.root
for num in tqdm(os.listdir(root), desc="Extract attributes"):
    prompt_value_dict = {}
    if not num.isdigit():
        continue
    os.makedirs(os.path.join(args.output, num), exist_ok=True)
    infer_command_path = os.path.join(root, num, "infer_command.json")
    infer_command = json.load(open(infer_command_path))
    
    # pred_labels as gold_labels
    infer_command['gold_labels'] = deepcopy(infer_command['pred_labels'])
    for idx, instrument in enumerate(I1s2):
        infer_command['gold_labels'][instrument] = infer_command['gold_labels']['I1s2'][idx]
    infer_command['gold_labels'].pop('I1s2')

    for idx, artist in enumerate(S4):
        infer_command['gold_labels'][artist] = infer_command['gold_labels']['S4'][idx]
    infer_command['gold_labels'].pop('S4')

    allmidiinfo[num] = deepcopy(infer_command)
    allinfermidivector[num] = {}
    midipath = os.path.join(root, num, "midi")
    all_midiname = os.listdir(midipath)
    # if len(all_midiname)>=10:
    #     need_process = random.sample(all_midiname, 10)
    # else:
    #     need_process = all_midiname
    # for midiname in need_process:
    for midiname in os.listdir(midipath):
        # print(midiname)
        extractor = DataExtractor(encoding_method='REMIGEN2', attribute_list_version='v3')
        try:
            att_dict = extractor.extract(
                midi_dir=midipath,
                midi_path=midiname,
                cut_method='none',
                pos_info_path=None,
                structure_func=None,
                emotion_func=None,
            )[3]["pieces"][0]["values"]
        except BaseException as e:
            print("Extracting Error: ")
            print(os.path.join(root, midipath, midiname))
            print(str(e))
            att_dict = None
        # print(att_dict)
        prompt_value_dict[midiname] = att_dict
        allinfermidivector[num][midiname] = {"value_dict": att_dict, "vector": {}, "acc": {}}
    json.dump(prompt_value_dict, open(os.path.join(args.output, num,"value_dict.json"),"w"))
    
# print("Write gold info")
# json.dump(allmidiinfo, open(os.path.join(args.output, "gold_info.json"), "w"))
# print("Write extracted value dicts")
# pickle.dump(allinfermidivector, open(os.path.join(args.output,"allinfer_valuedict.bin"), "wb"))

# allmidiinfo = json.load(open(os.path.join(args.output, "gold_info.json")))
# allinfermidivector = pickle.load(open(os.path.join(args.output,"allinfer_valuedict.bin"), "rb"))
for num, info in tqdm(allmidiinfo.items(), desc="Comput Accuracy"):
    for midiname in allinfermidivector[num].keys():        
        if allinfermidivector[num][midiname]['value_dict'] == None or len(allinfermidivector[num][midiname]['value_dict'])==0:
            allinfermidivector[num][midiname]['vector'] = None
            allinfermidivector[num][midiname]['acc'] = None
            allinfermidivector[num][midiname]['correct_num'] = 0
            allinfermidivector[num][midiname]['NA_num'] = 0
            allinfermidivector[num][midiname]['acc_ration'] = None
        else:
            gen_unit_dict = convert_value_dict_into_unit_dict(allinfermidivector[num][midiname]['value_dict'], midi_encoder)
            gen_vector = {}
            # get_vector_success = True
            # try:
            for att, value in allinfermidivector[num][midiname]['value_dict'].items():
                try:
                    if att in ["S2s1", "S4", "EM1"]:
                        gen_vector[att] = gen_unit_dict[att].get_vector(use=False)
                    elif value!=None and value!=(None, None):
                        gen_vector[att] = gen_unit_dict[att].get_vector(use=True)
                    else:
                        gen_vector[att] = gen_unit_dict[att].get_vector(use=False)
                except:
                    gen_vector[att] = None
            allinfermidivector[num][midiname]['vector'] = deepcopy(gen_vector)
            for idx, instrument in enumerate(I1s2):
                allinfermidivector[num][midiname]['vector'][instrument] = allinfermidivector[num][midiname]['vector']['I1s2'][idx]
            allinfermidivector[num][midiname]['vector'].pop('I1s2')
            for idx, artist in enumerate(S4):
                allinfermidivector[num][midiname]['vector'][artist] = None
            allinfermidivector[num][midiname]['vector'].pop('S4')
            allinfermidivector[num][midiname]['vector'].pop('I4')
            allinfermidivector[num][midiname]['vector'].pop('ST1')
            allinfermidivector[num][midiname]['vector'].pop('C1')

            cnt = 0
            allcnt = 0
            cntna = 0
            for att, vector in allinfermidivector[num][midiname]['vector'].items():
                if vector==None:
                    allinfermidivector[num][midiname]['acc'][att] = "NA"
                    cntna += 1
                    continue
                if att[:4]=="I1s2":
                    if info['gold_labels'][att][-1] == 1:
                        allinfermidivector[num][midiname]['acc'][att] = "NA"
                        cntna += 1
                    elif info['gold_labels'][att][1] == 1:
                        if vector[-1]==1 or vector[1]==1:
                            allinfermidivector[num][midiname]['acc'][att]=1
                            cnt += 1
                            allcnt += 1
                        else:
                            allinfermidivector[num][midiname]['acc'][att]=0
                            allcnt += 1
                    else:
                        if vector[0] == 1:
                            allinfermidivector[num][midiname]['acc'][att]=1
                            cnt += 1
                            allcnt += 1
                        else:
                            allinfermidivector[num][midiname]['acc'][att]=0
                            allcnt += 1
                elif att in ["S2s1", "S4", "EM1"]:
                    allinfermidivector[num][midiname]['acc'][att] = "NA"
                    cntna += 1
                elif att=="B1s1":
                    if info['gold_labels'][att][-1]==1 or vector[-1]==1:
                        allinfermidivector[num][midiname]['acc'][att] = "NA"
                        cntna += 1
                    else:
                        if vector == info['gold_labels'][att]:
                            allinfermidivector[num][midiname]['acc'][att] = 1
                            cnt += 1
                            allcnt += 1
                        else:
                            allinfermidivector[num][midiname]['acc'][att] = 0
                            allcnt += 1
                else:
                    if info['gold_labels'][att][-1] == 1:
                        allinfermidivector[num][midiname]['acc'][att] = "NA"
                        cntna += 1
                    else:
                        if vector == info['gold_labels'][att]:
                            allinfermidivector[num][midiname]['acc'][att] = 1
                            cnt += 1
                            allcnt += 1
                        else:
                            allinfermidivector[num][midiname]['acc'][att] = 0
                            allcnt += 1
            allinfermidivector[num][midiname]['correct_num'] = deepcopy(cnt)
            allinfermidivector[num][midiname]['NA_num'] = deepcopy(cntna)
            allinfermidivector[num][midiname]['attribute_num'] = deepcopy(allcnt)
            if allcnt!=0:
                allinfermidivector[num][midiname]['acc_ration'] = cnt / allcnt
            else:
                allinfermidivector[num][midiname]['acc_ration'] = None
            
    
allacc = {}
for key in key_order:
    allacc[key] = [0,0] # correct_num, all_num
for num, midiinfo in allinfermidivector.items():
    for midiname, infos in midiinfo.items():
        if infos['acc']==None:
            continue
        for att, ifacc in infos['acc'].items():
            if ifacc!="NA":
                allacc[att][1] += 1
            if ifacc == 1:
                allacc[att][0] += 1

for k in allacc.keys():
    if allacc[k][1] == 0:
        allacc[k].append(float("nan"))
    else:
        allacc[k].append(allacc[k][0] / allacc[k][1])
json.dump(allacc, open(os.path.join(args.output, f'acc_result.json'),'w'))
json.dump(allinfermidivector, open(os.path.join(args.output,f'midiinfo.json'),'w'))

ASA = 0.0
cnt = 0
for num in allinfermidivector.keys():
    for midiname in allinfermidivector[num].keys():
        if allinfermidivector[num][midiname]['acc_ration']:
            ASA += allinfermidivector[num][midiname]['acc_ration']
            cnt += 1
ASA = ASA / cnt
print("ASA:", ASA)
