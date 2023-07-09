import json
import random
import os
import itertools
from pathlib import Path
import pickle 
from tqdm.contrib.concurrent import process_map
import sys

fs = frozenset
opj = os.path.join
inst_label = "I1s2"

inst_class_id_to_inst_class_name = {
        # piano 0:
        0: ['piano', "grand piano"],

        # keyboard 1:
        1: ['keyboard', "digital keyboard", "synthesizer keyboard"],

        # percussion 2:
        2: ['percussion'],

        # organ 3:
        3: ['organ'],

        # guitar 4:
        4: ['guitar'],

        # bass 5:
        5: ['bass'],

        # violin 6:
        6: ['violin'],

        # viola 7:
        7: ['viola'],

        # cello 8:
        8: ['cello'],

        # harp 9:
        9: ['harp'],

        # strings 10:
        10: ['strings'],

        # voice 11:
        11: ['voice'],

        # trumpet 12:
        12: ['trumpet'],

        # trombone 13:
        13: ['trombone'],

        # tuba 14:
        14: ['tuba'],

        # horn 15:
        15: ['horn'],

        # brass 16:
        16: ['brass'],

        # sax 17:
        17: ['sax', "saxophone"],

        # oboe 18:
        18: ['oboe'],

        # bassoon 19:
        19: ['bassoon'],

        # clarinet 20:
        20: ['clarinet'],

        # piccolo 21:
        21: ['piccolo'],

        # flute 22:
        22: ['flute'],

        # pipe 23:
        23: ['pipe'],

        # synthesizer 24:
        24: ['synthesizer', "synth"],

        # ethnic instrument 25:
        25: ['ethnic instrument'],

        # sound effect 26:
        26: ['sound effect'],

        # drum 27:
        27: ['drum'],
    }

genre_label_to_name = {
    'New Age': "new age",
    'Electronic': "electronic",
    'Rap': 'rap',
    'Religious': 'religious',
    'International': 'international',
    'Easy_Listening': 'easy listening',
    'Avant_Garde': 'avant-garde',
    'RnB': 'RnB',
    'Latin': 'Latin',
    'Children': 'children',
    'Jazz': 'jazz',
    'Classical': 'classical',
    'Comedy_Spoken': 'comedy',
    'Pop_Rock': 'pop',
    'Reggae': 'reggae',
    'Stage': 'stage',
    'Folk': 'folk',
    'Blues': 'blues',
    'Vocal': 'vocal',
    'Holiday': 'holiday',
    'Country': 'country',
    "Symphony": 'symphony'
}

artist_label_to_artist_name = {
    'beethoven': 'Beethoven',
    'mozart': 'Mozart',
    'chopin': 'Chopin',
    'schubert': 'Schubert',
    'schumann': 'Schumann',
    'bach-js': 'Bach',
    'haydn': 'Haydn',
    'brahms': 'Brahms',
    'Handel': 'Handel',
    'tchaikovsky': 'Tchaikovsky',
    'mendelssohn': 'Mendelssohn',
    'dvorak': 'Dvorak',
    'liszt': 'Liszt',
    'stravinsky': 'Stravinsky',
    'mahler': 'Mahler',
    'prokofiev': 'Prokofiev',
    'shostakovich': 'Shostakovich',
}

label_to_template_v3 = {
    "I1":"[INSTRUMENTS]", 
    "I3":"[INSTRUMENT]", 
    "I4":"[INSTRUMENT]", 
    "C1":"[]",
    "R1":"[]",
    "R3":"[]",
    "S4":"[GENRE]", 
    "S2":"[ARTIST]", 
    "B1":"[NUM_BARS]", 
    "TS1":"[TIME_SIGNATURE]", 
    "K1":"[KEY]",
    "T1":"[]",
    "P4":"[RANGE]",
    "ST1":"[STRUCTURE]", 
    "EM1":"[EMOTION]", 
    "TM1":"[TM1]",
    "M1":"[INSTRUMENT]",
    "M2":"[]"
    }

v3_templates = json.load(open(os.path.join(os.path.dirname(__file__), "template.json")))
for label in v3_templates:
    v3_templates[label] = v3_templates[label].split(";")
    for i, _ in enumerate(v3_templates[label]):
        v3_templates[label][i] = _.strip()

_chatgpt_template = json.load(open(os.path.join(os.path.dirname(__file__), "refined_template.json"), "r"))
chatgpt_template = {}
for temp in _chatgpt_template:
    attributes = fs(temp["attributes"])
    response = temp["response"]

    if attributes not in chatgpt_template:
        chatgpt_template[attributes] = []
    chatgpt_template[attributes].append(response.strip())

def remove_digit(t):
    r = []
    for letter in t:
        if letter not in '0123456789':
            r.append(letter)
    r = ''.join(r)
    return r


class Verbalizer(object):
    def __init__(self):
        self.complete_temp = 0
        pass

    def instr_to_str(self, instr):
        res = []
        if isinstance(instr, int):
            instr = [instr]
        for _instr in instr:
            res.append(random.choice(inst_class_id_to_inst_class_name[_instr]))
        return self.concat_str(res)

    def attribute_to_str_v3(self, attribute_values):
        _attribute_values = {}
        for att in attribute_values:
            v = attribute_values[att]
            if v is None or v == (None, None) or v == [None, None]:
                continue
            if att == inst_label:
                if v[0]:
                    _attribute_values["I1_1"] = self.instr_to_str(v[0])
                if v[1]:
                    _attribute_values["I1_0"] = self.instr_to_str(v[1])
            if att == "I4":
                if v[0]:
                     _attribute_values["I4_1"] = self.instr_to_str(v[0])
            if att == "R3":
                _map = {0:0, 1:2, 2:1}
                _attribute_values["R3_"+str(_map[v])] = ""
            if att == "C1" or att == "R1":
                if v == True:
                    _attribute_values[att+"_1"] = ""
                elif v == False:
                    _attribute_values[att+"_0"] = ""
                else:
                    _attribute_values[att+"_"+str(v)] = ""
            if att == "P4":
                _attribute_values[att+"_1"] = str(v)
            if att == "K1":
                _attribute_values[att+"_1"] = v
            if att == "EM1":
                _attribute_values[att+"_1"] = self.emotion_to_str(v)
            if att == "S2s1":
                _attribute_values["S2_1"] = artist_label_to_artist_name[v]
            if att == "S4":
                if v[0]:
                    v_0 = [genre_label_to_name[_v] for _v in v[0]]
                    _attribute_values["S4_1"] = self.concat_str(v_0)
                if v[1]:
                    v_1 = [genre_label_to_name[_v] for _v in v[1]]
                    _attribute_values["S4_0"] = self.concat_str(v_1)
            if att == "B1s1":
                bar_range = [
                    ["1 ~ 4", "about 3", "about 2"], 
                    ["5 ~ 8", "about 7", "about 6"],
                    ["9 ~ 12", "about 11", "about 10"],
                    ["13 ~ 16", "about 15", "about 14"],
                        ["over 16"]
                    ]
                _attribute_values["B1_1"] = random.choice(bar_range[v[1]])
            if att == "TS1s1":
                if v == "other":
                    _attribute_values["TS1_o"] = ""
                else:
                    _attribute_values["TS1_1"] = f"{v[0]}/{v[1]}"
            if att == "T1s1":
                _map = {0:1, 1:2, 2:0}
                # _attribute_values["T1_"+str(_map[v])] = ""
                _attribute_values["T1_"+str(_map[v[1]])] = ""
            if att == "ST1":
                _attribute_values[att+"_1"] = remove_digit(v)
            if att == "TM1":
                time1 = [
                    ["1 ~ 15", "about 10"], 
                    ["16 ~ 30", "about 20"], 
                    ["31 ~ 45", "about 40"],
                    ["46 ~ 60", "about 50"], 
                    ["over 60"]]
                _attribute_values[att+"_1"] = random.choice(time1[v[1]])
        return _attribute_values

    def concat_str(self, str_list):
        str_list = [s for s in str_list if s != ""]
        if len(str_list) == 0:
            return ""
        if len(str_list) == 1:
            return str_list[0]
        res = str_list[0]
        if res[-1] == '.':
            res = res[:-1]
        for i in range(1, len(str_list) - 1):
            if str_list[i][-1] == ".":
                mid = str_list[i][1:-1]
            else:
                mid = str_list[i][1:]
            res += f", {str_list[i][0].lower()}{mid}"
        
        res += f" and {str_list[-1][0].lower()}{str_list[-1][1:]}"
        return res

    def emotion_to_str(self, emo):
        _emo = {
            "Q1": ["happiness", "excitement", 'joy', 'bliss', 'delight', 'elation', 'contentment', 'pleasure', 'satisfaction', 'cheerfulness', 'ecstasy', 'gladness', 'gratitude', 'jubilation', 'thrill', 'anticipation', 'exhilaration', 'adventure', 'stimulation', 'enthusiasm', 'euphoria', 'animation', 'zeal', 'fervor', 'verve', 'gusto'],
            "Q2": ["tension", "unease", 'anxiety', 'nervousness', 'apprehension', 'worry', 'distress', 'agitation', 'restlessness', 'jitters', 'uneasiness', 'disquiet', 'trepidation', 'insecurity', 'edginess', 'suspense', 'fearfulness', 'anticipation'],
            "Q3": ["sadness", "depression", 'grief', 'sorrow', 'despair', 'melancholy', 'misery', 'despondency', 'blues', 'heartache', 'regret', 'sullenness', 'mournfulness', 'dolefulness', 'dejection', 'hopelessness', 'pessimism', 'downheartedness'],
            "Q4": ["calmness", "relaxation", 'serenity', 'tranquility', 'peace', 'composure', 'ease', 'placidity', 'repose', 'quiet', 'stillness', 'restfulness', 'leisure', 'unwind', 'decompress', 'unwind', 'chill', 'out', 'rejuvenate'],
            1: ["happiness", "excitement", 'joy', 'bliss', 'delight', 'elation', 'contentment', 'pleasure', 'satisfaction', 'cheerfulness', 'ecstasy', 'gladness', 'gratitude', 'jubilation', 'thrill', 'anticipation', 'exhilaration', 'adventure', 'stimulation', 'enthusiasm', 'euphoria', 'animation', 'zeal', 'fervor', 'verve', 'gusto'],
            2: ["tension", "unease", 'anxiety', 'nervousness', 'apprehension', 'worry', 'distress', 'agitation', 'restlessness', 'jitters', 'uneasiness', 'disquiet', 'trepidation', 'insecurity', 'edginess', 'suspense', 'fearfulness', 'anticipation'],
            3: ["sadness", "depression", 'grief', 'sorrow', 'despair', 'melancholy', 'misery', 'despondency', 'blues', 'heartache', 'regret', 'sullenness', 'mournfulness', 'dolefulness', 'dejection', 'hopelessness', 'pessimism', 'downheartedness'],
            4: ["calmness", "relaxation", 'serenity', 'tranquility', 'peace', 'composure', 'ease', 'placidity', 'repose', 'quiet', 'stillness', 'restfulness', 'leisure', 'unwind', 'decompress', 'unwind', 'chill', 'out', 'rejuvenate']
        }
        _ = _emo[emo]
        return random.choice(_)

    def feeling_to_str(self, feel):
        _feel = {
            "F1": ["Bright", 'radiant', 'shining', 'luminous', 'vivid', 'brilliant', 'dazzling', 'beaming', 'glowing', 'sparkling', 'sunny', 'cheerful', 'optimistic', 'happy', 'joyful', 'lively', 'colorful'],
            "F2": ["Gloomy", 'melancholy', 'somber', 'depressing', 'miserable', 'dismal', 'bleak', 'desolate', 'sorrowful', 'morose', 'dark', 'dreary', 'funereal', 'disheartening', 'cheerless', 'despairing']
        }
        _ = _feel[feel]
        return random.choice(_)

    def get_combine_attributes(self, attribute_values):
        attributes = []
        remove_attr = []
        _attribute_values = {}
        for att in attribute_values:
            v = attribute_values[att]
            if att == "C1":
                if v == 0 or v == 1:
                    attributes.append(f"C1_{v}")
                    _attribute_values[f"C1_{v}"] = v
                else:
                    attributes.append("C1")
            elif att == inst_label and "EM1" in attribute_values:
                em1 = attribute_values["EM1"]
                attributes.append(f"I1_ALL_EM1_{em1}")
                _attribute_values[f"I1_ALL_EM1_{em1}"] = v
                remove_attr.append("EM1")
            elif att == inst_label and "ST1" in attribute_values:
                remove_attr.append("ST1")
                if v == "AA" or v == "AAAA":
                    attributes.append("I1_ST1_A")
                    _attribute_values["I1_ST1_A"] = v
                else:
                    attributes.append("I1_ST1")
                    _attribute_values["I1_ST1"] = v
            elif att == "K1" and "S2" in attribute_values:
                attributes.append("K1_S2")
                remove_attr.append("S2")
            else:
                attributes.append(att)
        
        for att in attribute_values:
            _attribute_values[att] = attribute_values[att]
        for att in remove_attr:
            attributes.remove(att)
        return attributes, _attribute_values

    def replace_template_with_attributes_v3(self, template, attribute, attribute_values):
        return template.replace(label_to_template_v3[attribute.split("_")[0]], attribute_values[attribute])

    def select_attributes_and_replace(self, attributes, attribute_values):
        for i in range(len(attributes)):
            l = len(attributes) - i
            if l == 1:
                break
            for it in itertools.combinations(attributes, l):
                attr_comb = fs([attr for attr in it])
                if attr_comb in chatgpt_template:                        
                    temp = random.choice(chatgpt_template[attr_comb])
                    failed = 0
                    for attr in attr_comb:
                        label = label_to_template_v3[attr.split("_")[0]]
                        if label != "[]" and label not in temp:
                            failed = 1
                            break
                    if failed == 1:
                        continue
                    
                    for attr in attr_comb:
                        temp = self.replace_template_with_attributes_v3(
                            template=temp, 
                            attribute=attr, 
                            attribute_values=attribute_values)
                    if "[" in temp or "]" in temp:
                        continue
                    return temp.strip(), attr_comb
        return None, None

    def get_text_from_chatgpt(self, attribute_values):
        attribute_values = self.attribute_to_str_v3(attribute_values)
        attributes = list(attribute_values.keys())
        res_strs = []
        random.shuffle(attributes)
        la = len(attributes)
        while len(attributes) >= 2:
            attr_comb = None
            temp, attr_comb = self.select_attributes_and_replace(attributes, attribute_values)
            
            if attr_comb is None:
                break

            if la == len(attr_comb):
                self.complete_temp += 1

            res_strs.append(temp)
            for attr in attr_comb:
                attributes.remove(attr)
        
        for attr in attributes:
            temp = random.choice(v3_templates[attr])
            res_strs.append(self.replace_template_with_attributes_v3(
                template=temp,
                attribute=attr,
                attribute_values=attribute_values))
        res_strs = [r.strip() for r in res_strs if r]
        res_strs = [r[0].upper() + r[1:] for r in res_strs]
        return " ".join(res_strs)

    def filter_template(self, text):
        text = text.strip().lower()
        scenery_sentence =text.count("suit") + text.count("perfect for") + text.count("scene")
        if (
            text[-1] != "." or 
            "lyric" in text or 
            "video" in text or 
            scenery_sentence >= 2 or
            "paragraph" in text
            ):
            return False
        return True
    
    def get_text(self, attribute_values, mode="chatgpt", retry = 10):
        """
        返回所有文本的list
        :param attribute_values: dict, key是attribute标号, value是attribute_unit的extract函数返回的信息
        :param mode: 生成文本的方法, chatgpt是多attributes组合模板
        :return:
        """
        if mode == "chatgpt":
            for i in range(retry):
                text = self.get_text_from_chatgpt(attribute_values).strip()
                if text and self.filter_template(text):
                    return text
            return False

    def run(self, v):
        try:
            if "use_info_dict" in v:
                text = self.get_text(v["use_info_dict"], retry=100)
                if not text:
                    return v
                v["text"] = text
                return v
            elif inst_label in v:
                v["text"] = self.get_text(v)
                return v
        except Exception as e:
            return v
    
if __name__ == "__main__":
    bin_folder = sys.argv[1]

    verbalizer = Verbalizer()

    mode = "chatgpt"
    bin_path_list = list(Path(opj(os.path.dirname(__file__), bin_folder)).glob("*.bin"))
    save_path = opj(os.path.dirname(__file__), f"{bin_folder}_refined")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    restore_run = False
    refined = 0
    all_temp = 0
    step = 12000

    # restore only support one task
    if restore_run:
        save_bin_path_list = list(Path(
            opj(os.path.dirname(__file__), save_path)).glob("*.bin"))
        save_bin_path_list = sorted(
            save_bin_path_list, 
            key=lambda x: int(x.name.split("_")[0]), 
            reverse=True)
        restore_from = save_bin_path_list[0].name.split("_")[0]
        restore_from = int(restore_from) + step 
        human_test = []
        with open(save_bin_path_list[0], "rb") as f, open(bin_path_list[0], "rb") as ff:
            human_test = pickle.load(f)

            human_value_dict = pickle.load(ff)[restore_from:]

            human_test.extend(
                process_map(
                    verbalizer.run, 
                    human_value_dict, 
                    max_workers=12, 
                    chunksize=100))
            failed_sample = [
                v for v in human_test if "text" not in v or v["text"] == ""
                ]
            print(len(failed_sample) / len(human_test))

            human_test = [v for v in human_test if "text" in v and v["text"] != ""]

            pickle.dump(human_test, open(opj(save_path, "full_"+bin_path_list[0].name), "wb"))
                
        exit()

    else:
        for bin_path in bin_path_list:
            human_test = []
            value_dict_list = []
            with open(bin_path, "rb") as f:
                human_value_dict = pickle.load(f)
                for i in range(0, len(human_value_dict), step):
                    value_dict_list += [human_value_dict[i:i+step]]
            for i, value_dict in enumerate(value_dict_list):
                human_test.extend(
                    process_map(
                        verbalizer.run, 
                        value_dict, 
                        max_workers=12, 
                        chunksize=100))
                failed_sample = [
                    v for v in human_test if "text" not in v or v["text"] == ""
                    ]

                print(failed_sample)

                print(len(failed_sample) / len(human_test))

                human_test = [v for v in human_test if "text" in v and v["text"] != ""]

                pickle.dump(human_test, open(opj(save_path, str(i * step)+"_"+bin_path.name), "wb"))
                
                pickle.dump(failed_sample, open(opj(save_path, "failed_sample.bin"), "wb"))
