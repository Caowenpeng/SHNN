import pyedflib
import numpy as np
import h5py
import csv
import math
import pandas as pd
import os


sc_data_path = '../SleepEdfData/sleep-cassette/'
sc_data_files = os.listdir(sc_data_path)

st_data_path = '../SleepEdfData/sleep-telemetry/'
st_data_files = os.listdir(st_data_path)

sc_h5file = "../SleepEdfData/SCDataSet/"
sc_h5_files = os.listdir(sc_h5file)

st_h5file = "../SleepEdfData/STDataSet/"
st_h5_files = os.listdir(st_h5file)

channels = ['Fpz-Cz', 'Pz-Oz']

def saveLabelFile(file_name, contents):
    df = pd.DataFrame(contents)
    df.to_csv(file_name, index=None, header=None)

##Getting the label corresponding to the data
def getLabel(annotations1, annotations2):
    row = 0
    label = []
    annotations2error = []
    for annotation1 in annotations1:
        if annotations2[row] == 'Sleep stage W':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 0
        elif annotations2[row] == 'Sleep stage R':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 4
        elif annotations2[row] == 'Sleep stage 1':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 1
        elif annotations2[row] == 'Sleep stage 2':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 2
        elif annotations2[row] == 'Sleep stage 3':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 3
        elif annotations2[row] == 'Sleep stage 4':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 3
        elif annotations2[row] == 'Sleep stage ?':
            annotation1 = 0
            ##Used for statistics except for the last stage? Beyond, stage unknown data
            if row != (len(annotations1) - 1):
                annotations2error.append(row)
        else:
            annotation1 = 0
            annotations2error.append(row)
        for index in range(annotation1):
            label.extend(annotations2[row])
        row += 1
    return label, annotations2error

##Removing data with abnormal labels
def deleteOtherData(datas, annotations2error, annotations0, annotations1):
    deletelist = []
    for index in annotations2error:

        delete_index = np.arange(int(float(annotations0[index])) * 100,
                                 (int(float(annotations0[index])) + int(float(annotations1[index]))) * 100,
                                 dtype=int)
        deletelist.extend(delete_index)

    datas = np.delete(datas, deletelist)
    return datas

##Determining whether the length and number of data points are consistent
def judgementDataLen(datasecond, datasize):

    if int(float(datasecond)) * 100 == datasize:
        return True
    else:
        return False

##Determining whether the length of data points and the number of sleep stages are consistent
def getTrueannotations1(annotations0, annotations1):
    for index in range(len(annotations0) - 1):
        annotations1[index] = int(float(annotations0[index + 1])) - int(float(annotations0[index]))
    annotations1[index + 1] = int(float(annotations1[index + 1]))
    return annotations1

###Getting label data in EDF file
def getChannelData(datafile, labelfile, savefile, savelabelfile, channelsNum, sample_rate):
    fileedf = pyedflib.EdfReader(datafile)
    filelabel = pyedflib.EdfReader(labelfile)
    signal_headers = fileedf.getSignalHeaders()
    channelsName = []
    channelsData = []
    for index in range(channelsNum):
        channel_name = signal_headers[index]['label'][4:]
        channelsName.append(channel_name)
        column = sample_rate * 30  #Number of columns for a 30 second epoch
        size = fileedf.readSignal(index, 0, None, False).size  # Getting the total number of data points

        ##Getting events
        annotations = np.array(filelabel.readAnnotations())
        ##Modifying by the number of data points
        if judgementDataLen(annotations[0][-1], size) != True:
            annotations[0][-1] = size // 100

        annotations[1] = getTrueannotations1(annotations[0], annotations[1])
        labels, annotations2error = getLabel(annotations[1], annotations[2])
        ##Preventing the first installment from being less than half an hour in duration
        if int(float(annotations[0][1]) * sample_rate) > 30 * 80 * sample_rate:
            start = int(float(annotations[0][1]) * sample_rate) - 30 * 80 * sample_rate
        else:
            start = int(float(annotations[0][0]))
        end = size
        oldsignals = fileedf.readSignal(index, 0, end, False)
        if len(annotations2error) != 0:
            oldsignals = deleteOtherData(oldsignals, annotations2error, annotations[0], annotations[1])
        signals = oldsignals[start:]
        channelsData.append(signals.reshape(-1, column))
    with h5py.File(savefile, 'w') as fileh5:
        fileh5['sample_rate'] = np.array([sample_rate], dtype=int)
        for channelindex in range(len(channelsData)):
            fileh5[channelsName[channelindex]] = channelsData[channelindex].reshape(-1, column)[:1080, :]
        labels = np.array(labels, dtype='int32').reshape(-1, 1)[:1080, :]
        saveLabelFile(savelabelfile, labels)
        fileh5['label'] = labels


def cutData(x_data, y_data, size):
    len = x_data.shape[0] // size * size
    x_data = x_data[:len, :]
    y_data = y_data[:len, :]

    return x_data, y_data

##Reshaping the data as a batch
def shuffleData(x_data, y_data, size):


    x_data = x_data.reshape(-1, size, 3000)
    y_data = y_data.reshape(-1, size, 1)

    return x_data, y_data

##Reshaping the data according to a batch
def noshuffleData(x_data, y_data):
    x_data = x_data.reshape(-1, 3000)
    y_data = y_data.reshape(-1, 1)

    return x_data, y_data

##Extract data from EDF file and put into H5 file
if __name__ == '__main__':

    file_indexs = np.array(range(len(sc_data_files)))[::2]
    file_num = 0
    channelsData = [[], [], []]

    #Extracting the data of the sc dataset
    for file_index in file_indexs:
        getChannelData(sc_data_path + sc_data_files[file_index], sc_data_path + sc_data_files[file_index + 1], sc_h5file + "data/" + str(file_num + 1).zfill(4) + ".h5",
                       sc_h5file + "label/" + str(file_num + 1).zfill(4), 3, 100)

        file_num += 1



