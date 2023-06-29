import random

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
import warnings
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
# Import libraries
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import lightgbm as lgb

# 从数据集中选取feature的阈值
def EMOPIA_threshold():
    feature_table = np.load(r"../../StyleCtrlData/jSymbolic_lib\datasets\EMOPIA\data\feature_table.npy")
    X, Y = feature_table[:, :-1], feature_table[:, -1]

    # for i in range(X.shape[1]):
    #     medium_value = np.percentile(X[:, i], 50)
    #     thresholds.append(medium_value)
    #     X_discrete.append(np.searchsorted([medium_value], X[:, i])[:, np.newaxis])
    for bucket_num in [3, 5, 8, 12, 16]:
        thresholds = []
        for i in range(X.shape[1]):
            thres = []
            for j in range(1, bucket_num):
                thres.append(np.percentile(X[:, i], 100.0/bucket_num*j))
            thresholds.append(thres)
            # X_discrete.append(np.searchsorted(thres, X[:, i])[:, np.newaxis])
        thresholds = np.array(thresholds)
        # X_discrete = np.concatenate(X_discrete, axis = 1)
        # cv_folds = 5
        # skf = StratifiedKFold(n_splits=cv_folds, random_state=2022, shuffle=True)
        # y_preds = np.zeros(Y.shape) - 1
        # for train_index, test_index in skf.split(X_discrete, Y):
        #     X_train, y_train = X_discrete[train_index], Y[train_index]
        #     X_test, y_test = X_discrete[test_index], Y[test_index]
        #     clf = RandomForestClassifier(n_estimators=100, random_state=2022)
        #     clf.fit(X_train, y_train)
        #     y_pred_score = clf.predict_proba(X_test)
        #     y_pred = np.argmax(y_pred_score, axis=1)
        #     y_pred += 1  # 预测第几个象限
        #     y_preds[test_index] = y_pred
        # print(f"离散化后的{X_discrete.shape[1]}维特征:", f1_score(Y, y_preds, average="micro"))
        np.save(f"../../StyleCtrlData/jSymbolic_lib/datasets/EMOPIA/data/threshold_{bucket_num}.npy", thresholds)
    # 二值化之后的分类精度为0.6762523191094619

    # feature_selected = np.load("./data/selected_feature/emotion_And_select_17.npy")
    # feature_name = np.load("./data/feature_name.npy")
    # name2index = dict(zip(feature_name, range(len(feature_name))))
    # feature_selected_index = [name2index[i] for i in feature_selected]
    # X_discrete = X_discrete[:, feature_selected_index]
    # y_preds = np.zeros(Y.shape) - 1
    # for train_index, test_index in skf.split(X_discrete, Y):
    #     X_train, y_train = X_discrete[train_index], Y[train_index]
    #     X_test, y_test = X_discrete[test_index], Y[test_index]
    #     clf = RandomForestClassifier(n_estimators=100, random_state=2022)
    #     clf.fit(X_train, y_train)
    #     y_pred_score = clf.predict_proba(X_test)
    #     y_pred = np.argmax(y_pred_score, axis=1)
    #     y_pred += 1  # 预测第几个象限
    #     y_preds[test_index] = y_pred
    # print(f"离散化后的{X_discrete.shape[1]}维特征:", f1_score(Y, y_preds, average="micro"))
    # 分为两个桶：
        # 离散化后的1495维特征: 0.6762523191094619
        # 离散化后的39维特征: 0.62430426716141
        # 离散化后的17维特征: 0.5092764378478665
    # 分为三个桶
        # 离散化后的1495维特征: 0.6725417439703154
        # 离散化后的39维特征: 0.6447124304267161
        # 离散化后的17维特征: 0.5445269016697588
def VGMIDI_feedback():
    threshold = np.load("./datasets/EMOPIA/data/threshold_2.npy")
    feature_table = np.load("./datasets/VGMIDI/data/feature_table.npy")
    X = feature_table[:, :-1]
    Y = feature_table[:, -1]
    X_discrete = []
    for i in range(X.shape[1]):
        X_discrete.append(np.searchsorted([threshold[i]], X[:,i])[:, np.newaxis])
    X_discrete = np.concatenate(X_discrete, axis = 1)
    cv_folds = 5
    skf = StratifiedKFold(n_splits=cv_folds, random_state=2022, shuffle=True)
    y_preds_continuous = np.zeros(Y.shape) - 1
    for train_index, test_index in skf.split(X, Y):
        X_train, y_train = X[train_index], Y[train_index]
        X_test, y_test = X[test_index], Y[test_index]
        clf = RandomForestClassifier(n_estimators=100, random_state=2022)
        clf.fit(X_train, y_train)
        y_pred_score = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred_score, axis=1)
        y_pred += 1  # 预测第几个象限
        y_preds_continuous[test_index] = y_pred
    print(f1_score(Y, y_preds_continuous, average="micro"))

def TopMAGD_threshold():
    feature_table = []
    labels = []
    path_root = r"../..\StyleCtrlData\data\1004_TopMAGD\truncated_2560\split_data"
    for split in ["train","valid", "test"]:
        feature_table.append(np.load(path_root + f"/1495/{split}_raw_command_1495.npy"))
        labels.append(np.load(path_root + f"/{split}_style_labels.npy"))
    feature_table = np.vstack(feature_table)
    labels = np.vstack(labels)
    X = feature_table
    Y = labels

    feature_selected = np.load(r"E:\Music\Project\StyleCtrl\StyleCtrlData\jSymbolic_lib\data\style_select_feature\style_or_10_select_103.npy")
    feature_names = np.load(r"E:\Music\Project\StyleCtrl\StyleCtrlData\jSymbolic_lib\data\feature_name.npy")
    feature2id = dict(zip(feature_names, range(len(feature_names))))
    select_features = [feature2id[i] for i in feature_selected]
    id2genre = np.load(r"E:\Music\Project\StyleCtrl\StyleCtrlData\data\1004_TopMAGD\truncated_2560\split_data\id2genre.npy")
    X = X[:, select_features]

    # 不离散化的feature
    cv_folds = 5
    skf = StratifiedKFold(n_splits=cv_folds, random_state=2022, shuffle=True)
    y_preds = np.zeros(Y.shape) - 1
    for index, style_name in enumerate(id2genre):
        Y_style_label = Y[:, index]
        for train_index, test_index in skf.split(X, Y_style_label):
            X_train, y_train = X[train_index], Y_style_label[train_index]
            X_test, y_test = X[test_index], Y_style_label[test_index]
            clf = RandomForestClassifier(n_estimators=100, random_state=2022)
            clf.fit(X_train, y_train)
            y_pred_score = clf.predict_proba(X_test)
            y_pred = np.argmax(y_pred_score, axis=1)
            y_pred += 1  # 预测第几个象限
            y_preds[test_index] = y_pred
        print(f"连续的{X.shape[1]}维特征:", style_name, f1_score(Y_style_label, y_preds, average="micro"), roc_auc_score(Y_style_label, y_preds))

    X_discrete = []
    for bucket_num in [2, 3, 5, 8, 12, 16]:
        thresholds = []
        for i in range(X.shape[1]):
            thres = []
            for j in range(1, bucket_num):
                thres.append(np.percentile(X[:, i], 100.0 / bucket_num * j))
            thresholds.append(thres)
            X_discrete.append(np.searchsorted(thres, X[:, i])[:, np.newaxis])
        # thresholds = np.array(thresholds)
        X_discrete = np.concatenate(X_discrete, axis = 1)
        cv_folds = 5
        skf = StratifiedKFold(n_splits=cv_folds, random_state=2022, shuffle=True)
        y_preds = np.zeros(Y.shape) - 1
        for train_index, test_index in skf.split(X_discrete, Y):
            X_train, y_train = X_discrete[train_index], Y[train_index]
            X_test, y_test = X_discrete[test_index], Y[test_index]
            clf = RandomForestClassifier(n_estimators=100, random_state=2022)
            clf.fit(X_train, y_train)
            y_pred_score = clf.predict_proba(X_test)
            y_pred = np.argmax(y_pred_score, axis=1)
            y_pred += 1  # 预测第几个象限
            y_preds[test_index] = y_pred
        print(f"离散化后的{X_discrete.shape[1]}维特征:", f1_score(Y, y_preds, average="micro"), roc_auc_score(Y, y_preds))
        # np.save(path_root + f"/threshold_{bucket_num}.npy", thresholds)
    # 二值化之后的分类精度为0.6762523191094619

    # feature_selected = np.load("./data/selected_feature/emotion_And_select_17.npy")
    # feature_name = np.load("./data/feature_name.npy")
    # name2index = dict(zip(feature_name, range(len(feature_name))))
    # feature_selected_index = [name2index[i] for i in feature_selected]
    # X_discrete = X_discrete[:, feature_selected_index]
    # y_preds = np.zeros(Y.shape) - 1
    # for train_index, test_index in skf.split(X_discrete, Y):
    #     X_train, y_train = X_discrete[train_index], Y[train_index]
    #     X_test, y_test = X_discrete[test_index], Y[test_index]
    #     clf = RandomForestClassifier(n_estimators=100, random_state=2022)
    #     clf.fit(X_train, y_train)
    #     y_pred_score = clf.predict_proba(X_test)
    #     y_pred = np.argmax(y_pred_score, axis=1)
    #     y_pred += 1  # 预测第几个象限
    #     y_preds[test_index] = y_pred
    # print(f"离散化后的{X_discrete.shape[1]}维特征:", f1_score(Y, y_preds, average="micro"))

def YMDB_threshold():
    feature_table = np.load(r"../../StyleCtrlData/jSymbolic_lib/datasets/1224_YMDB/data/fea_table.npy")
    X = feature_table
    # X, Y = feature_table[:, :-1], feature_table[:, -1]

    # for i in range(X.shape[1]):
    #     medium_value = np.percentile(X[:, i], 50)
    #     thresholds.append(medium_value)
    #     X_discrete.append(np.searchsorted([medium_value], X[:, i])[:, np.newaxis])
    for bucket_num in [2, 3, 5, 8, 12, 16]:
        thresholds = []
        for i in range(X.shape[1]):
            thres = []
            for j in range(1, bucket_num):
                thres.append(np.percentile(X[:, i], 100.0 / bucket_num * j))
            thresholds.append(thres)
            # X_discrete.append(np.searchsorted(thres, X[:, i])[:, np.newaxis])
        thresholds = np.array(thresholds)
        np.save(f"../../StyleCtrlData/jSymbolic_lib/datasets/1224_YMDB/data/threshold_{bucket_num}.npy", thresholds)


if __name__ == "__main__":
    # EMOPIA_threshold()
    # TopMAGD_threshold()
    YMDB_threshold()