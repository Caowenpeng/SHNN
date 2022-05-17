import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import os
import torch
import h5py
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F
from ResultCode import *
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from Utils import *

train_Batch_Size = 24
BATCH_SIZE = 120
use_gpu = torch.cuda.is_available()   #Determining if GPU is available
h5file = "./data/"
files = os.listdir(h5file)
files_len = len(files)
keys = ["Fpz-Cz", "Pz-Oz", "label"]

def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()

## Getting data and label from h5 files
def getEEGData(h5file, filesname, channel):
    data = np.empty(shape=[0, 3000])
    labels = np.empty(shape=[0, 1])
    for filename in filesname:
        with h5py.File(h5file + filename, 'r') as fileh5:
            data = np.concatenate((data, fileh5[keys[channel]].value), axis=0)
            labels = np.concatenate((labels, fileh5[keys[2]].value), axis=0)
    data = (torch.from_numpy(data)).type('torch.FloatTensor')
    labels = (torch.from_numpy(labels)).type('torch.LongTensor')
    labels = labels.squeeze(dim=1)
    return data, labels

###Normalizing and return t he normalized data and standardscaler
def standardScalerData(standardscaler, x_data):
    standardscaler.fit(x_data)
    x_standard = standardscaler.transform(x_data)
    return torch.from_numpy(x_standard), standardscaler

#Test the model effect
def testModel(shnn, x_test, y_test, standardscaler, index, all_label, all_pred, result_path):
    shnn.eval()
    torch_dataset_test = Data.TensorDataset(x_test, y_test.squeeze())
    data_loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=True)
    if use_gpu:
        with torch.no_grad():
            pred_y_list = []
            label_y_list = []
            for step, (test_x, test_y) in enumerate(data_loader_test):
                x_test_standard = standardscaler.transform(test_x)
                x_test_standard = torch.from_numpy(x_test_standard)
                test_x = torch.unsqueeze(x_test_standard, 1).type(torch.FloatTensor).cuda()
                shnn = shnn.cuda()
                output = shnn(test_x)
                test_pred_y = torch.max(output, 1)[1].cpu()
                pred_y_list.extend(test_pred_y)
                label_y_list.extend(test_y)
        all = []
        all.append(label_y_list)
        all.append(pred_y_list)
        all_label.extend(label_y_list)
        all_pred.extend(pred_y_list)
        ##Saving the test set data and Calculating various evaluation indicators
        saveLabelFile(result_path + "label/%d.csv" % (index + 1), np.array(all).T)
        kappa, classification_report_result, cm = cm_plot_number(np.array(label_y_list).squeeze(), np.array(pred_y_list).squeeze(),
                 "_kfold-%s" % (str(index + 1)), "%s/matrix/" % (result_path))
        np.savetxt('%s/kappa/kappa_%s.txt' % (result_path, str(index + 1)), [kappa])
        saveExcelFile("%s/matrix/%s.csv" % (result_path, str(index + 1)), cm)
        saveExcelFile("%s/report/report_%s.csv" % (result_path, str(index + 1)), classification_report_result)
        torch.cuda.empty_cache()
        return all_label, all_pred