import numpy as np
import os
import torch
import h5py
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, classification_report
from ResultCode import *
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from Utils import *

PRE_EPOCH = 40
T1_EPOCH = 40
T2_EPOCH = 60
A = 0.1
train_Batch_Size = 24
BATCH_SIZE = 120

use_gpu = torch.cuda.is_available()   #判断GPU是否存在可用
keys = ["Fpz-Cz", "Pz-Oz", "label"]
result_path = "./result/"
##Saving scaler
scaler_path = './result/scaler/'
scaler_files = os.listdir(scaler_path)
savepseudofile = './pseudodata/'

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

##Iterative training
def trainModle(modlename, lossname, optimizername, data_loader, EPOCH, index):
    standardscaler = StandardScaler()
    for epoch in range(EPOCH):
        modlename.train()
        for step, (train_x, train_y) in enumerate(data_loader):
            train_x, train_y = noshuffleData(train_x, train_y)
            ##Normalizing
            train_x, standardscaler = standardScalerData(standardscaler, train_x)
            ##Putting the model, data and loss function into the GPU
            if use_gpu:
                modlename = modlename.cuda()
                lossname = lossname.cuda()
                train_x = torch.unsqueeze(train_x, 1).type(torch.FloatTensor).cuda()
                train_y = torch.squeeze(train_y).type(torch.LongTensor).cuda()
            output = modlename(train_x)
            loss_train = lossname(output, train_y)
            optimizername.zero_grad()  # clear gradients for this training step
            loss_train.backward()  # backpropagation, compute gradients
            optimizername.step()
    # Saving the normalizer
    pickle.dump(standardscaler, open(scaler_path + "%s.pkl" % str(index + 1), 'wb'))
    torch.cuda.empty_cache()
    torch.save(modlename.state_dict(), result_path + 'module/%d.pth' % (index + 1))  #entire net
    return modlename, standardscaler

###Pseudo-label training of the model using unlabel data
def pseudoTrainModel(modlename, lossname, optimizername, data_loader, pseudo_x_all, standardscaler, PRE_EPOCH, EPOCH, index):
    for epoch in range(PRE_EPOCH, EPOCH):
        pseudo_data_loader = getLabelFromExcel(modlename, index, pseudo_x_all)
        modlename.train()
        for step, (train_data, pseudo_train_data) in enumerate(zip(data_loader, pseudo_data_loader)):  # 设定训练数据
            #Getting batch-sized label and unlabel data
            train_x = train_data[0]
            train_y = train_data[1]
            pseudo_x = pseudo_train_data[0]
            pseudo_y = pseudo_train_data[1]
            train_x, train_y = noshuffleData(train_x, train_y)
            pseudo_x, pseudo_y = noshuffleData(pseudo_x, pseudo_y)

            # Concatenate label data and unlabel data
            train_pseudo_x = torch.cat((train_x, pseudo_x), dim=0)
            # Putting the model, data and loss function into the GPU
            if use_gpu:
                train_pseudo_x, standardscaler = standardScalerData(standardscaler, train_pseudo_x)
                train_pseudo_x = torch.unsqueeze(train_pseudo_x, 1).type(torch.FloatTensor).cuda()#先 为数据加上通道维度，并改为float类型
                train_y = torch.squeeze(train_y).type(torch.LongTensor).cuda()
                pseudo_y = torch.squeeze(pseudo_y).type(torch.LongTensor).cuda()
                modlename.cuda()
                lossname.cuda()
            output = modlename(train_pseudo_x)
            # As the number of iterations ofmodel training increases, the weights of the label loss and the pseudo-label loss keep changing
            a = A * (epoch + 1 - PRE_EPOCH) / (T2_EPOCH - T1_EPOCH)
            loss_train = lossname(output[:train_x.shape[0]], train_y) + lossname(output[train_x.shape[0]:], pseudo_y) * a
            optimizername.zero_grad()  # clear gradients for this training step
            loss_train.backward()  # backpropagation, compute gradients
            optimizername.step()
    return modlename, standardscaler

##Using a pre-model to get pseudo-label and putting them into the dataloader
def getLabelFromExcel(shnn, index, pseudo_x):
        ##获取伪标签
        pseudo_y = getPseudoLabel(shnn, index, pseudo_x)
        pseudo_x, pseudo_y = cutData(pseudo_x[:pseudo_y.shape[1], :], pseudo_y, size=5)
        pseudo_x, pseudo_y = shuffleData(pseudo_x, pseudo_y, size=5)
        torch_dataset_pseudo_train = Data.TensorDataset(pseudo_x, torch.from_numpy(pseudo_y).squeeze())
        pseudo_data_loader = Data.DataLoader(dataset=torch_dataset_pseudo_train, batch_size=train_Batch_Size // 3 * 2, shuffle=True, drop_last=True)
        return pseudo_data_loader

##Using the pre-model to predict without labels and get pseudo-label
def getPseudoLabel(shnn, index, x_pseudo):
    shnn.eval()
    scaler_files = os.listdir(scaler_path)
    scaler = pickle.load(open(scaler_path + scaler_files[index], 'rb'))
    torch_dataset_pseudo_train = Data.TensorDataset(x_pseudo)
    pseudo_data_loader = Data.DataLoader(dataset=torch_dataset_pseudo_train, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    if use_gpu:
        with torch.no_grad():
            pseudo_y_list = []
            for step, (pseudo_x) in enumerate(pseudo_data_loader):
                pseudo_x = pseudo_x[0]
                x_pseudo_standard = scaler.transform(pseudo_x)
                x_pseudo_standard = torch.from_numpy(x_pseudo_standard)
                pseudo_x = torch.unsqueeze(x_pseudo_standard, 1).type(torch.FloatTensor).cuda()
                shnn = shnn.cuda()
                output = shnn(pseudo_x)
                pseudo_pred_y = torch.max(output, 1)[1].cpu()
                pseudo_y_list.extend(pseudo_pred_y)
            pseudo_y_list = np.array(pseudo_y_list, dtype='int32').reshape(1, -1)
    return pseudo_y_list



