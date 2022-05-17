import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.datasets import load_breast_cancer
import os
from Model import *
import torch
import torch.nn as nn
import h5py
import torch.utils.data as Data
from ResultCode import *
from Utils import *
from SHNNTrain import *

EPOCH = 100
PRE_EPOCH = 40
T1_EPOCH = 40
T2_EPOCH = 60
A = 0.1

BATCH_SIZE = 24
LEARNING_RATE = 0.0001
use_gpu = torch.cuda.is_available()

h5file = "./data/"
files = os.listdir(h5file)
files_len = len(files)
keys = ["Fpz-Cz", "Pz-Oz", "label"]
modelPath = './result/module/'
model_files = os.listdir(modelPath)
model_files_len = len(model_files)
kfold = 5

## Train SHNN Model
if __name__ == '__main__':

    kf = KFold(n_splits=5)
    index = 0
    eeg_data, labels = getEEGData(h5file, files, channel=0)
    for train_index, test_index in kf.split(eeg_data):
        x_train, x_test = eeg_data[train_index], eeg_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        ## Divide the training set data into label data and label data
        x_train_split = x_train[:len(x_train) // 3]
        y_train_split = y_train[:len(y_train) // 3]
        pseudo_x = x_train[len(x_train)//3:]
        pseudo_y = y_train[len(y_train)//3:]
        x_train_split_cut, y_train_split_cut = cutData(x_train_split, y_train_split, size=5)
        x_train_split_shuffle, y_train_split_shuffle = shuffleData(x_train_split_cut, y_train_split_cut, size=5)
        torch.cuda.empty_cache()
        if (use_gpu):
            shnn = SHNN()
            optimizer = torch.optim.Adam(shnn.parameters(), lr=LEARNING_RATE)
            loss_func = nn.CrossEntropyLoss()
            torch_dataset_train = Data.TensorDataset(x_train_split_shuffle, y_train_split_shuffle.squeeze())
            data_loader = Data.DataLoader(dataset=torch_dataset_train, batch_size=BATCH_SIZE, shuffle=True,
                                          drop_last=True)
            ##Step 1: Pre-training with label data
            shnn, standardscaler = trainModle(shnn, loss_func, optimizer, data_loader, PRE_EPOCH, index)
            torch_dataset_train = Data.TensorDataset(x_train_split_shuffle, y_train_split_shuffle.squeeze())
            data_loader = Data.DataLoader(dataset=torch_dataset_train, batch_size=BATCH_SIZE // 3, shuffle=True,
                                          drop_last=True)
            ##Step 2: Add unlabel data for training
            shnn, standardscaler = pseudoTrainModel(shnn, loss_func, optimizer, data_loader, pseudo_x, pseudo_y,
                                                    standardscaler, PRE_EPOCH, EPOCH, index, x_test, y_test)

        index += 1






