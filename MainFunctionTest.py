
import numpy as np
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import os
from SHNN import *
import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import h5py
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F
from ResultCode import *
##from Attention import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from Utils import *
from FocalLoss import *
from SHNNTest import *
import librosa

import mne
from mne.time_frequency import psd_multitaper





LEARNING_RATE = 0.0001

use_gpu = torch.cuda.is_available()   #判断GPU是否存在可用
h5file = "./data/"
files = os.listdir(h5file)  # 得到文件夹下的所有文件名称
files_len = len(files)
result_path = "./result/"
##K折交叉验证的数量
kfold = 5
keys = ["Fpz-Cz", "Pz-Oz", "label"]

modelPath = './result/module/'
model_files = os.listdir(modelPath)  # 得到文件夹下的所有文件名称
model_files_len = len(model_files)

standardscalerPath = './result/scaler/'
standardscaler_files = os.listdir(standardscalerPath)  # 得到文件夹下的所有文件名称
standardscaler_files_len = len(standardscaler_files)

##排序
def sort_filterout(out):
    final_out = []
    stage_number = [0]
    stage_number_list = [0]
    for stage_index in range(out.shape[0]):
        for filter_index in range(out.shape[1]):
            if np.argmax(out[:, filter_index]) == stage_index:
                final_out.append(out[:, filter_index])
        stage_number.append(len(final_out))
        stage_number_list.append(len(final_out) - stage_number_list[stage_index])
    return final_out, stage_number, stage_number_list[1:]

if __name__ == '__main__':

    kf = KFold(n_splits=5)
    index = 0

    eeg_data, labels = getEEGData(h5file, files)
    all_label = []
    all_pred = []

    for train_index, test_index in kf.split(eeg_data):
        x_test = eeg_data[test_index]
        y_test = labels[test_index]
        model = SHNN()
        model.load_state_dict(torch.load(modelPath + model_files[index]))
        standardscaler = pickle.load(open(standardscalerPath + standardscaler_files[index], 'rb'))
        ##测试模型
        all_label, all_pred = testModel(model, x_test, y_test, standardscaler, index, all_label, all_pred, result_path)

        index += 1

    all = []
    all.append(all_label)
    all.append(all_pred)
    ##保存测试集数据
    saveLabelFile(result_path + "/all/label.csv", np.array(all).T)
    kappa, classification_report_result, cm = cm_plot_number(np.array(all_label).squeeze(),
                                                             np.array(all_pred).squeeze(),
                                                             "all",
                                                             "%s/all/" % (result_path))
    np.savetxt('%s/all/kappa.txt' % (result_path), [kappa])
    cm_report = []
    saveExcelFile("%s/all/cm.csv" % (result_path), cm)
    saveExcelFile("%s/all/report.csv" % (result_path), classification_report_result)












