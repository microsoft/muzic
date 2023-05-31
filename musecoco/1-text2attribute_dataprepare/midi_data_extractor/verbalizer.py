import json
import random
import os
from music21 import pitch
from .const import inst_class_id_to_name_2
# from const import inst_class_id_to_name

_inst_class_id_to_name = inst_class_id_to_name_2

inst_label = "I1s1"
inst_c_label = "I2s1"
metas = ["B1", "TS1", "T1", "P1", "P2", "P3", "C2", "R1", "R2", "S1", "S3", "P4", "TS1s1","B1s1","K1"]
label_to_template = {
    inst_label:"[INSTRUMENTS]", inst_c_label : "[INSTRUMENT]",
    "B1": "[NUM_BARS]", "B1s1" : "[NUM_BARS]", "TS1": "[TIME_SIGNATURE]", "TS1s1" :"[TIME_SIGNATURE]",
    "T1": "[TEMPO]", "P1": "[LOW_PITCH]", "P2": "[HIGH_PITCH]", "P3": "[NUM_PITCH_CLASS]",
    "ST1": "[STRUCTURE]", "EM1": "[EMOTION]","C2":"[]", "R1":"[]", "R2":"[]", "S1":"[]","S3":"[]",
    "K1": "[KEY]", "S2":"[ARTIST]", "P4":"[RANGE]"
    }


def remove_digit(t):
    r = []
    for letter in t:
        if letter not in '0123456789':
            r.append(letter)
    r = ''.join(r)
    return r


class Verbalizer(object):
    def __init__(self):
        pass

    def instr_to_str(self, instr):
        res = []
        for _instr in instr:
            res.append(_inst_class_id_to_name[_instr])
        return self.concat_str(res)

    def attribute_to_str(self, attribute_values):
        _attribute_values = {}
        for att in attribute_values:
            v = attribute_values[att]
            if v is None or v == (None, None):
                continue
            if (att == "C2" or att == "R1" or att == "R2" or att == "S1" or att == "S3") and v == 1:
                _attribute_values[att] = "True"
            if att == "B1" or att == "P4":
                _attribute_values[att] = str(v)
            if att == "B1s1":
                bar_range = ["1-4", "5-8", "9-12", "13-16"]
                _attribute_values[att] = bar_range[v[1]]
            if att == "P1" or att == "P2":
                _attribute_values[att] = pitch.Pitch(midi=v).nameWithOctave
            if att == "TS1":
                _attribute_values[att] = f"{v[0]}/{v[1]}"
            if att == "T1":
                _attribute_values[att] = v[1]
            if att == "P3":
                # attribute_values[att] = str(v)[1:-1]
                _attribute_values[att] = str(len(v))
            if att == inst_label:
                _attribute_values[att] = self.instr_to_str(v)
            if att == inst_c_label:
                if v[1] is not None:
                    _attribute_values[att] = (v[0], self.instr_to_str(v[1]))
                if v[2] is not None:
                    _attribute_values[att] = (v[0], self.instr_to_str(v[2]))
            if att == "ST1":
                _attribute_values[att] = remove_digit(v)
            if att == "EM1" or att == "K1" or att == "S2" or att == "C1":
                _attribute_values[att] = v
            if att == "TS1s1":
                if len(v) == 2:
                    _attribute_values[att] = f"{v[0]}/{v[1]}"
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
            "Q1": ["happiness", "excitement"],
            "Q2": ["tension ", " unease"],
            "Q3": ["sadness ", " depression"],
            "Q4": ["calmness ", " relaxation"]
        }
        _ = _emo[emo]
        return random.choice(_)

    def feeling_to_str(self, feel):
        _feel = {
            "F1": ["Bright"],
            "F2": ["Gloomy"]
        }
        _ = _feel[feel]
        return random.choice(_)

    def get_text(self, attribute_values):
        """
        返回所有文本的list
        :param attribute_values: dict, key是attribute标号，value是attribute_unit的extract函数返回的信息
        :return:
        """
        # TODO 单复数模板
        # TODO 专有名词大小写
        template = json.load(open(os.path.join(os.path.dirname(__file__), "template.txt"), "r"))
        for label in template:
            template[label] = template[label].split(";")
            for i, _ in enumerate(template[label]):
                template[label][i] = _.strip()
        instr_all = attribute_values[inst_label]
        attribute_values = self.attribute_to_str(attribute_values)

        meta_info = [meta for meta in metas if meta in attribute_values]
        meta_str_with_info = []
        random_meta = 10
        max_meta = 4
        for i in range(random_meta):
            _meta_info = random.sample(meta_info, random.randint(1, min(max_meta, len(meta_info))))

            meta_strs = []
            for meta in _meta_info:
                meta_temp = random.choice(template[meta])
                try:
                    meta_strs.append(meta_temp.replace(label_to_template[meta], attribute_values[meta]))
                except:
                    print(meta)
                    raise
            concat_meta_strs = self.concat_str(meta_strs)
            meta_str_with_info.append((_meta_info, concat_meta_strs))

        core_strs = []
        
        if "EM1" in attribute_values:
            v = attribute_values["EM1"]
            for temp in template["I1_ALL_EM1_" + v]:
                core_strs.append((["EM1", "I1_ALL"], temp.replace("[INSTRUMENT]", attribute_values[inst_label])))

            if len(instr_all) > 1:
                instr = random.sample(instr_all, 1)
                instr_name = self.instr_to_str(instr)
                for temp_em in template["EM1"]:
                    for temp_i in template[inst_label]:
                        temp_em = temp_em.replace("[EMOTION]", self.emotion_to_str(v))
                        temp_i = temp_i.replace("[INSTRUMENTS]", instr_name)
                        core_strs.append((["EM1", inst_label, instr], self.concat_str([temp_em, temp_i])))

        if "ST1" in attribute_values:
            v = attribute_values["ST1"]
            if v == "AA" or v == "AAAA":
                _id = "I1_ST1_A"
            else:
                _id = "I1_ST1"

            for temp in template[_id]:
                _temp = temp.replace("[INSTRUMENT]", attribute_values[inst_label])
                core_strs.append((["ST1", "I1_ALL"], _temp.replace("[STRUCTURE]", v)))
        
        if "I1s1" in attribute_values:
            for temp in template[inst_label]:
                _temp = temp.replace("[INSTRUMENTS]", attribute_values[inst_label])
                core_strs.append((["I1_ALL"], _temp))

        mid_strs = []
        # instr_var = "

        if "K1" in attribute_values and "S2" in attribute_values:
            K1_S2_temp = template["K1_S2"]
            K1_S2_temp = random.choice(K1_S2_temp)
            K1_S2_temp = K1_S2_temp.replace(label_to_template["K1"], attribute_values["K1"])
            K1_S2_temp = K1_S2_temp.replace(label_to_template["S2"], attribute_values["S2"])
            mid_strs.append(["K1_S2",K1_S2_temp])
        
        if "C1" in attribute_values:
            v = attribute_values["C1"]
            if v == 0 or v == 1:
                temp = random.choice(template[f"C1_{v}"])
            elif v == 2:
                temp = random.choice(template["C1"])
                temp = temp.replace("[FEELING_A]", self.feeling_to_str("F1"))
                temp = temp.replace("[FEELING_B]", self.feeling_to_str("F2"))
            elif v == 3:
                temp = random.choice(template["C1"])
                temp = temp.replace("[FEELING_A]", self.feeling_to_str("F2"))
                temp = temp.replace("[FEELING_B]", self.feeling_to_str("F1"))
            mid_strs.append(["C1", temp])

        if "S2" in attribute_values and "K1" not in attribute_values:
            temp = random.choice(template["S2"])
            S2_temp = temp.replace(label_to_template["S2"], attribute_values["S2"])
            mid_strs.append(["S2", S2_temp])
        
        if inst_c_label in attribute_values:
            if attribute_values[inst_c_label][0] == "inc":
                instr_var = "[INSTRUMENT] is added in the middle."
            else:
                instr_var = "[INSTRUMENT] is removed in the middle."
            instr_var = instr_var.replace("[INSTRUMENT]", attribute_values[inst_c_label][1])
            instr_var = instr_var[0].upper() + instr_var[1:]
            mid_strs.append([inst_c_label, instr_var])
        
        # mid_strs = random.choices(mid_strs, k=min(3, len(mid_strs)))
        mid_attr = [att[0] for att in mid_strs]
        mid_strs = [s[1] for s in mid_strs]
        mid_str = self.concat_str(mid_strs)

        core_strs = random.sample(core_strs, 4)
        meta_str_with_info = random.sample(meta_str_with_info, 4)
        
        res_str = []
        used_attr = []
        for i in range(len(core_strs)):
            for j in range(len(meta_str_with_info)):
                attr = {}
                for _meta in meta_str_with_info[j][0]:
                    attr[_meta] = 1
                for _i, _meta in enumerate(core_strs[i][0]):
                    if _meta == "I1_ALL":
                        attr[inst_label] = instr_all
                        break
                    if _meta == inst_label:
                        attr[inst_label] = tuple(core_strs[i][0][_i + 1])
                        break
                    attr[_meta] = 1
                for att in mid_attr:
                    if att == "K1_S2":
                        attr["K1"] = 1; attr["S2"] = 1
                    else:
                        attr[att] = 1
                if mid_str:
                    s = " ".join([core_strs[i][1], mid_str, meta_str_with_info[j][1]])
                else:
                    s = " ".join([core_strs[i][1], meta_str_with_info[j][1]])
                res_str.append(s)
                used_attr.append(attr)

        return res_str, used_attr


if __name__ == "__main__":
    verbalizer = Verbalizer()
    test_attr = {inst_label: set([0, 1]), inst_c_label: ("inc", None, set([3, 4]), 4), "ST1": None, "B1": 16, "TS1": (3, 4),
                 "T1": (108.0, "Moderato"), "P1": 50, "P2": 70, "P3": set([1, 2, 3, 4]), "EM1": "Q1"}
    test_none_attr = {inst_label: set([0, 1]), inst_c_label: None, "ST1": None, "B1": 16, "TS1": None,
                 "T1": (None, None), "P1": 50, "P2": 70, "P3": set([1, 2, 3, 4]), "EM1": "Q1"}
    no_emo_and_st_attr = {inst_label: set([0, 1]), inst_c_label: None, "ST1": None, "B1": 16, "TS1": None,
                 "T1": (None, None), "P1": 50, "P2": 70, "P3": set([1, 2, 3, 4]), "EM1": None}
    test_attr_2 = {inst_label: {16}, inst_c_label: ('inc', {16}, None, 1), 'B1': 16, 'TS1': (4, 4), 'T1': (None, None), 'P1': None, 'P2': None, 'P3': None, 'ST1': None, 'EM1': 'Q3'}
    
    v2_test = {inst_label: set([0, 1]), inst_c_label: ("inc", None, set([3, 4]), 4), "ST1": None, "B1": 16, "TS1": (3, 4),
                 "T1": (108.0, "Moderato"), "P1": 50, "P2": 70, "P3": set([1, 2, 3, 4]), "EM1": "Q1", 
                 "C1": 0, "C2": 1, "R1": 1, "R2": 1, "S1":1, "S2":None, "K1": "major",
                 "P4": 3, "S3":1 ,"I1s1":{0,1}, "I2s1": ("inc", None, set([3, 4]), 4) }
    v2_test_P4 = {inst_label: set([0, 1]), inst_c_label: ("inc", None, set([3, 4]), 4), "ST1": None, "B1": None, "TS1": None,
                 "T1": (None, None), "P3": None, "EM1": "Q1", 
                 "P4": 3, "S3":1 ,"K1": "major"}
    v2_test_B1s1_TS1s1 = {inst_label: set([0, 1]), inst_c_label: None, "ST1": None,
                 "T1": (None, None), "P3": None, "EM1": "Q1", 
                 "C1": None, "C2": None, "R1": None, "R2": None, "S1":None, "S2": None, "K1": None, "B1s1":(16, 3), "TS1s1":(3, 4)}
    test_none_attr = {inst_label: set([0, 1]), inst_c_label: None, "ST1": None, "B1": 16, "TS1": None,
                 "T1": (None, None), "P1": 50, "P2": 70, "P3": None, "EM1": "Q1", 
                 "C1": None, "C2": None, "R1": None, "R2": None, "S1":None, "S2": None, "K1": None}
    
    v2_test_2 = {'I1s1': {0, 3, 4, 13}, 'I2s1': ('inc', {3, 4, 13}, None, 14), 'C1': 2, 'R2': False, 'S1': None, 'S2': None, 'S3': None, 'B1s1': (16, 3), 'TS1s1': (4, 4), 'K1': 'minor', 'T1': (120.81591202325676, 'Allegro'), 'P3': {0, 2, 4, 5, 7, 9, 11}, 'P4': 5, 'ST1': None, 'EM1': 'Q2'}
    v2_test_3 = {'I1s1': {0}, 'I2s1': ('inc', {0}, None, 2), 'C1': 3, 'R2': False, 'S1': None, 'S2': None, 'S3': None, 'B1s1': (16, 3), 'TS1s1': (4, 4), 'K1': 'minor', 'T1': (120.81591202325676, 'Allegro'), 'P3': {0, 2, 4, 5, 7, 9, 11}, 'P4': 3, 'ST1': None, 'EM1': 'Q4'}

    print("=====Start Test====")
    verbalizer.get_text(test_attr)
    verbalizer.get_text(test_attr_2)
    verbalizer.get_text(test_none_attr)
    verbalizer.get_text(no_emo_and_st_attr)
    verbalizer.get_text(v2_test)
    verbalizer.get_text(v2_test_P4)
    verbalizer.get_text(v2_test_B1s1_TS1s1)
    verbalizer.get_text(v2_test_2)
    verbalizer.get_text(v2_test_3)
    verbalizer.get_text(test_none_attr)
    print("=====Passed====")