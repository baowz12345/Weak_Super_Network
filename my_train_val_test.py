#!/usr/bin/env python
# encoding: utf-8
from os.path import dirname, abspath, exists

import numpy as np
import argparse, h5py
import os

from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.model_selection import KFold
from my_utils import AllSample, SelectBest, PlotandSave, Id_k_folds
from my_models import RNNls, WSCNNLSTMwithNoisy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta, SGD
def parse_args():
    parser = argparse.ArgumentParser(description="Convert sequence and target")
    parser.add_argument('-datalable', dest='datalable', type=str, help='Data for training, testing',default=r'D:\workship\anewlife\wscnnlstmchange\rawdata\H1sec0'
                                                                                                            r'\datalabel.hdf5')
    parser.add_argument('-k', dest='k_folds', type=int, default=3, help='k-folds cross-validation')
    parser.add_argument('-batchsize', dest='batchsize', type=int,default=128,help='the size of one batch')
    parser.add_argument('-ratio', dest='ratio', type=float, default=0.2, help='the propotion of validation data over the training data')
    parser.add_argument('-params', dest='params', type=int,default=15, help='the number of paramter settings')
    parser.add_argument('-train', dest='train', action='store_true', default=True, help='only test step')
    parser.add_argument('-plot', dest='plot', action='store_true', default=True, help='only test step')

    return parser.parse_args()


def main():
    file_path = dirname(abspath(__file__))
    args = parse_args()

    #打印出正确的数据集
    # split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
    name = (args.datalable).split('\\')[-3]
    print(('working on is %s now' %name))



    #转换数据
    print('begin to encoding.')
    with h5py.File(args.datalable, 'r') as f:
        seqs_vector = np.asarray(f['data'])
        labels = np.asarray(f['label'])

    print("data loaded")
    seqs_num = seqs_vector.shape[0];
    seqs_len = seqs_vector.shape[1]
    seqs_dim = seqs_vector.shape[2]
    print(('there are %d seqences, each of which is a %d*%d array' % (seqs_num, seqs_len, seqs_dim)))
    input_shape = (seqs_len, seqs_dim)

    # print("data loaded")
    # seqs_num = seqs_vector.shape[0]
    # instance_num = seqs_vector.shape[1]
    # instance_len = seqs_vector.shape[2]
    # instance_dim = seqs_vector.shape[3]
    # print(('there are %d seqences, each of which is a %d*%d*%d array' % (
    # seqs_num, instance_num, instance_len, instance_dim)))
    # input_shape = (instance_num, instance_len, instance_dim)

    indices = np.arange(seqs_num)
    np.random.shuffle(indices)
    seqs_vector = seqs_vector[indices]
    labels = labels[indices]

    train_ids, test_ids, valid_ids = Id_k_folds(seqs_num, args.k_folds, args.ratio)
    roc_auc1 = []
    pr_auc1 = []
    f1score = []
    a1 = []
    space = AllSample()

    if not exists(r'D:\workship\anewlife\wscnnlstmchange\model\my_test'):
        print('Building D:\workship\anewlife\wscnnlstmchange\model\my_test')
        os.makedirs(r'D:\workship\anewlife\wscnnlstmchange\model\my_test')
    f_params = open(r'D:\workship\anewlife\wscnnlstmchange\model\my_test\params.txt', 'w')

    #i=0
    for fold in range(args.k_folds):
        X_train = seqs_vector[train_ids[fold]]
        y_train = labels[train_ids[fold]]
        X_test = seqs_vector[test_ids[fold]]
        y_test = labels[test_ids[fold]]
        X_valid = seqs_vector[valid_ids[fold]]
        y_valid = labels[valid_ids[fold]]



        if args.train:
            history_all = {}
            for params_num in range(args.params):
                params = space[params_num]
                print(("the {}-th paramter setting of the {}-th is {}".format(params_num, fold, params)))
                print("the {}-th paramter setting of the {}-th is {}".format(params_num, fold, params), file=f_params)

                print ('Building model...')
                model = DanQ(input_shape, params)
                checkpointer = ModelCheckpoint(filepath = r'D:\workship\anewlife\wscnnlstmchange\model\my_test\params%d_bestmodel_%dfold.hdf5' %( params_num, fold), monitor='val_accuracy', verbose=1, save_best_only=True)
                earlystopper = EarlyStopping(monitor='val_accuracy', patience=6, verbose=1)

                print('Training model...')
                # myoptimizer = SGD(decay=params['DELTA'], momentum=params['MOMENT'], learning_rate=0.01)
                myoptimizer = Adadelta(epsilon=params['DELTA'], rho=params['MOMENT'], lr = 1)
                model.compile(loss='binary_crossentropy', optimizer=myoptimizer, metrics=['accuracy'])
                History = model.fit(X_train, y_train, epochs=60, batch_size=args.batchsize, shuffle=True, validation_data=(X_valid, y_valid), callbacks=[checkpointer, earlystopper], verbose=2)
                history_all[str(params_num)] = History

            best_num = SelectBest(history_all, file_path + '/model/%s/' % name, fold, 'val_accuracy')
            if args.plot:
                PlotandSave(history_all[str(best_num)], r'D:\workship\anewlife\wscnnlstmchange\model\my_test\figure_%dfold.png' % (fold), fold, 'val_accuracy')

        print('Begin testing model...')
        model.load_weights(r'D:\workship\anewlife\wscnnlstmchange\model\my_test\params%d_bestmodel_%dfold.hdf5' % ( best_num, fold))

        results = model.evaluate(X_test, y_test)
        print('result:', results)
        y_pred = model.predict(X_test, batch_size=args.batchsize, verbose =1)
        y_pred = np.asarray([y[0] for y in y_pred])
        y_real = np.asarray([y[0] for y in y_test])


        #将结果写入文件
        with open(r'D:\workship\anewlife\wscnnlstmchange\model\my_test\score_%dfold.txt' %(fold), 'w') as f:
            assert len(y_pred) == len(y_real), 'dismathed!'
            for i in range(len(y_pred)):
                print('{:.5f} {}'.format(y_pred[i], y_real[i]), file = f)

        print('Calculating AUC...')


        ##accuracy
        a = accuracy_score(y_real,y_pred.round())
        a1.append(a)
        #rocauc
        fpr, tpr ,thresholds = roc_curve(y_real, y_pred)
        roc_auc  = auc(fpr, tpr)
        roc_auc1.append(roc_auc)
        #prauc
        precision, recall, thresholds = precision_recall_curve(y_real,y_pred)
        pr_auc = auc(recall,precision)
        pr_auc1.append(pr_auc)
        #f1score
        b = f1_score(y_real,y_pred.round())
        f1score.append(b)

    f_params.close()
    print(("the average rocauc is {} and the average prauc is {}".format(np.mean(roc_auc1), np.mean(pr_auc1))))
    print("the average f1score is {}".format(np.mean(f1score)))
    print("the average accuracy is {}".format(np.mean(a1)))
    outfile = r'D:\workship\anewlife\wscnnlstmchange\model\my_test\metrics.txt'
    with open(outfile,'w') as f:
        for i in range(len(roc_auc1)):
            print("{:.5f} {:.5f} {:.5f} {:.5f}".format(roc_auc1[i], pr_auc1[i], f1score[i], a1[i]), file=f)
        print("{:.5f} {:.5f} {:.5f} {:.5f}".format(np.mean(roc_auc1), np.mean(pr_auc1), np.mean(f1score), np.mean(a1)), file=f)


if __name__ == '__main__':
    main()



