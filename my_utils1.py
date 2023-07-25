#!/usr/bin/env python
# encoding: utf-8

import numpy as np, sys, math, os, h5py
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from multiprocessing import Pool
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def Id_k_folds(seqs_num, k_folds, ratio):
    train_ids = [];
    test_ids = [];
    valid_ids = []
    if k_folds == 1:
        train_num = int(seqs_num * 0.7)
        test_num = seqs_num - train_num
        valid_num = int(train_num * ratio)
        train_num = train_num - valid_num
        index = range(seqs_num)
        train_ids.append(np.asarray(index[:train_num]))
        valid_ids.append(np.asarray(index[train_num:train_num + valid_num]))
        test_ids.append(np.asarray(index[train_num + valid_num:]))
    else:
        each_fold_num = int(math.ceil(seqs_num / k_folds))
        for fold in range(k_folds):
            index = range(seqs_num)
            index_slice = index[fold * each_fold_num:(fold + 1) * each_fold_num]
            index_left = list(set(index) - set(index_slice))
            test_ids.append(np.asarray(index_slice))
            train_num = len(index_left) - int(len(index_left) * ratio)
            train_ids.append(np.asarray(index_left[:train_num]))
            valid_ids.append(np.asarray(index_left[train_num:]))

    return (train_ids, test_ids, valid_ids)


#画出与保存训练过程
def PlotandSave(History, filepath, fold, moitor='val_loss'):
    if moitor == 'vol_loss':
        train_loss = History.history['loss']
        vaild_loss = History.history['val_loss']
        x = range(len(train_loss))

        plt.figure(num=fold)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.plot(x, train_loss, 'r-', x, vaild_loss, 'g-')
        plt.legend(['train_loss', 'vaild_loss'], loc = 'upper left')

    else:
        train_acc = History.history['accuracy']
        vaild_acc = History.history['val_accuracy']
        x=range(len(train_acc))

        plt.figure(num=fold)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.plot(x, train_acc, 'r-', x, vaild_acc, 'g-')
        plt.legend(['train_acc', 'vaild_acc'], loc = 'upper left')

    plt.savefig(filepath, format = 'png')



def AllSample():
    DROPOUT = [0.1, 0.5, 0.75]
    DELTA = [1e-04, 1e-06, 1e-08]
    MOMENT = [0.9, 0.99, 0.999]
    space = []

    for drop in DROPOUT:
        for delta in DELTA:
            for moment in MOMENT:
                space.append({'DROPOUT': drop, 'DELTA': delta, 'MOMENT': moment})
    return space


#选择最佳参数
def SelectBest(history_all, file_path, fold, monitor='val_loss'):
    if monitor == 'val_loss':
        loss  = 100000.
        for num, History in history_all.items():
            if np.min(History.history['val_loss']) < loss:
                best_num = int(num)
                loss = np.min(History.history['val_loss'])
    else:
        accuracy = 0.
        #items() 函数以列表返回可遍历的(键, 值) 元组数组。
        for num, History in history_all.items():
            if np.min(History.history['val_accuracy']) > accuracy:
                best_num = int(num)
                accuracy = np.max(History.history['val_accuracy'])

    del_num = list(range(len(history_all)))
    del_num.pop(best_num)
    #删除无用的模型参数
    for num in del_num:
        os.remove(r'D:\workship\anewlife\wscnnlstmchange\model\my_test1\params%d_bestmodel_%dfold.hdf5' %(num, fold))
    return  best_num
