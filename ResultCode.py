
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from copy import deepcopy
import matplotlib as plt
import os
import h5py
from sklearn import preprocessing
import torch.utils.data as Data
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
from sklearn.model_selection import KFold
import copy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import ast

## Calculation kappa
def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)

## Drawing confusion matrix
def cm_plot_number(original_label, predict_label, knum, savepath):
    cm = confusion_matrix(original_label, predict_label)
    cm_new = np.zeros(shape=[5, 5])
    for x in range(5):
        t = cm.sum(axis=1)[x]
        for y in range(5):
            cm_new[x][y] = round(cm[x][y] / t * 100, 2)

    plt.matshow(cm_new, cmap=plt.cm.Blues)

    plt.colorbar()
    x_numbers = []
    y_numbers = []
    for x in range(5):
        y_numbers.append(cm.sum(axis=1)[x])
        x_numbers.append(cm.sum(axis=0)[x])
        for y in range(5):
            percent = format(cm_new[x, y] * 100 / cm_new.sum(axis=1)[x], ".2f")

            plt.annotate(format(cm_new[x, y] * 100 / cm_new.sum(axis=1)[x], ".2f"), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center', fontsize=10)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')

    y_stage = ["W\n(" + str(y_numbers[0]) + ")", "N1\n(" + str(y_numbers[1]) + ")", "N2\n(" + str(y_numbers[2]) + ")",
               "N3\n(" + str(y_numbers[3]) + ")", "REM\n(" + str(y_numbers[4]) + ")"]
    x_stage = ["W\n(" + str(x_numbers[0]) + ")", "N1\n(" + str(x_numbers[1]) + ")", "N2\n(" + str(x_numbers[2]) + ")",
               "N3\n(" + str(x_numbers[3]) + ")", "REM\n(" + str(x_numbers[4]) + ")"]
    y = [0, 1, 2, 3, 4]
    plt.xticks(y, x_stage)
    plt.yticks(y, y_stage)
    plt.tight_layout()
    plt.savefig("%smatrix%s.svg" % (savepath, str(knum)), bbox_inches='tight')
    plt.show()
    plt.close()
    return kappa(cm), classification_report(original_label, predict_label, digits=6, output_dict=True), cm

def saveReportFile(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()

##Saving the Excel file and specify the number of decimal places
def saveExcelFile(file_name, contents):
    df = pd.DataFrame(contents).transpose().round(4)
    df.to_csv(file_name)

def save_data(knum, filenum,label,pred):
    test_pred_path = "D:/WorkSpace/Shell/MNN/result_new20/test_pred_files/"  # 数据路径
    filename1 = test_pred_path + str(knum) + "_test_pred_" + str(filenum) + ".h5"
    f1 = h5py.File(filename1, 'w')
    f1['label'] = label
    f1['result'] = pred
    f1.close()

def plot_sleepPicture(label, predict, savepath, filename):
    label_new = replace_stage(copy.deepcopy(label))
    predict_new = replace_stage(copy.deepcopy(predict))
    plot_sleep_label_pred(label_new, predict_new, savepath, filename + "label_pred.svg")
    plot_sleep(label_new, savepath, filename + "label.svg", "red")
    plot_sleep(predict_new, savepath, filename + "pred.svg", "blue")



##Replacing the number corresponding to the sleep stage
def replace_stage(old_array):
    old_array[old_array == 4] = 5
    old_array[old_array == 3] = 4
    old_array[old_array == 2] = 3
    old_array[old_array == 1] = 2
    old_array[old_array == 0] = 1
    return old_array

## Drawing a night sleep chart
def plot_sleep(label, savepath, filename, color):
    x = range(0, len(label))
    y = np.arange(5)
    y_stage = ["W", "N1", "N2", "N3", "REM"]
    plt.figure(figsize=(16, 5))
    plt.ylabel("Sleep Stage")
    plt.xlabel("30s Epoch(120 epochs = 1 hour)")
    plt.yticks(y, y_stage)
    plt.xlim(0, len(label))
    plt.plot(x, label, linestyle='-', color=color, alpha=1, linewidth=1)
    plt.legend(loc='best')
    plt.savefig("%s/%s" % (savepath, filename))
    plt.show()
    plt.close()

## Drawing a night sleep chart
def plot_sleep_label_pred(label, pred, savepath, filename):
    x = range(0, len(label))
    y = np.arange(5)
    y_stage = ["W", "REM", "N1", "N2", "N3"]
    plt.figure(figsize=(16, 5))
    plt.ylabel("Sleep Stage")
    plt.xlabel("Sleep Time")
    plt.yticks(y, y_stage)
    plt.xlim(0, len(label))
    plt.plot(x, label, linestyle='-', color='red', alpha=1, linewidth=1, label='label')
    plt.plot(x, pred, linestyle='-', color='blue', alpha=1, linewidth=1, label='predict')
    plt.legend(loc='best')
    plt.savefig("%s/%s" % (savepath, filename))
    plt.close()
    plt.show()

def plotLineChart(feature, path):
    x = range(feature.shape[0])
    plt.figure(figsize=(16, 8))
    plt.plot(x, feature,
             color='green',
             alpha=0.9,
             linewidth=1)
    plt.savefig(path)
