import os
import xmltodict
import numpy as np
from tqdm import tqdm
nan_feature_list = {}
split_name = ["Melodic Interval Histogram", "Vertical Interval Histogram"]
end_at = "Initial Time Signature"
# split_pos = [['Melodic Interval Histogram', 190], ['Vertical Interval Histogram', 342]]

feature_name = []
histogram_feature_name = []
def read_pitch_feature_without_long_histogram(path):
    data = xmltodict.parse(open(path, "r").read())
    data = data['feature_vector_file']["data_set"]["feature"]
    ret = []
    histogram_ret = []
    for f in data:
        if f["name"] == end_at:
            break
        if "histogram" not in f["name"].lower():
            if f["v"] == "NaN":
                ret.append(0)
                if f["name"] not in nan_feature_list.keys():
                    nan_feature_list[f["name"]] = 1
                else:
                    nan_feature_list[f["name"]] += 1
            elif isinstance(f["v"], list):
                ret.extend([eval(i) for i in f["v"]])
            else:
                ret.append(eval(f["v"]))
                # feature_name.append(f["name"])
        else:
            if len(f["v"]) < 20:
                # histogram_feature_name.append(f["name"])
                histogram_ret.extend([eval(i) for i in f["v"]])

    return ret, histogram_ret

def read_pitch_feature(path):
    data = xmltodict.parse(open(path, "r").read())
    data = data['feature_vector_file']["data_set"]["feature"]
    ret = []
    for f in data:
        # print(f["name"])
        # if f["name"] in split_name:
        #     split_pos.append([f["name"], len(ret)])
        if f["name"] == end_at:
            break
        if "histogram" not in f["name"].lower():
            if f["v"] == "NaN":
                ret.append(0)
                if f["name"] not in nan_feature_list.keys():
                    nan_feature_list[f["name"]] = 1
                else:
                    nan_feature_list[f["name"]] += 1
            elif isinstance(f["v"], list):
                ret.extend([eval(i) for i in f["v"]])
            else:
                ret.append(eval(f["v"]))
        else:
            ret.extend([eval(i) for i in f["v"]])

    return ret

def read_all_feature(path, need_name = False):
    data = open(path, "r").read()
    if len(data) == 0:
        return None
    data = xmltodict.parse(open(path, "r").read())
    data = data['feature_vector_file']["data_set"]["feature"]
    ret = []
    feature_names = []
    for f in data:
        if "histogram" not in f["name"].lower():
            if f["v"] == "NaN":
                ret.append(0)
                if need_name:
                    feature_names.append(f["name"])
                if f["name"] not in nan_feature_list.keys():
                    nan_feature_list[f["name"]] = 1
                else:
                    nan_feature_list[f["name"]] += 1
            elif isinstance(f["v"], list):
                ret.extend([eval(i) for i in f["v"]])
                if need_name:
                    feature_names.extend(f["name"] + f"_{i}" for i in range(len(f["v"])))
            else:
                ret.append(eval(f["v"]))
                if need_name:
                    feature_names.append(f["name"])
        else:
            ret.extend([eval(i) for i in f["v"]])
            if need_name:
                feature_names.extend(f["name"] + f"_{i}" for i in range(len(f["v"])))
    if need_name:
        assert len(ret) == len(feature_names)
        return ret, feature_names
    else:
        return ret
