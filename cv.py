#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)
import sys
import numpy as np
import pandas as pd
import pickle
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import set_session
import time
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import Callback

sys.path.append('models')
from CNN import cnn_v1,cnn_v2,rnn_v1,rcnn_v1
from RNN import rnn_att,rnn_att2
sys.path.append('utils/')
import config

from help import score, train_batch_generator, train_test, get_X_Y_from_df
from CutWord import cut_word,read_cut
from Data2id import data2id
from sklearn.metrics import accuracy_score

def load_data():
    data = read_cut(config.origin_csv,config.train_data_cut_hdf)
    data = data2id(data)
    x_train, y_train = get_X_Y_from_df(data)
    
    
    return x_train, y_train


def make_train_cv_data(X_train, Y_train, Model, model_name, epoch_nums, kfolds,lr):

    from keras.models import model_from_json

    json_string = Model.to_json()

    S_train = np.zeros((Y_train.shape[0], epoch_nums))
    S_Y = np.zeros((Y_train.shape[0], 1))

    train_df = pd.DataFrame()
    X, Y = X_train, Y_train
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=kfolds, shuffle=True,random_state=2018)
    k = 0
       
    epoch_nums =1 
    a = []
    for train_index, test_index in kf.split(Y):
        k += 1
        model = model_from_json(json_string)
        model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
        K.set_value(model.optimizer.lr, lr)
        for epoch_num in range(epoch_nums):   
            x_train = X[train_index, :]
            x_dev = X[test_index, :]
            y_train=Y[train_index,:]
            y_dev = Y[test_index, :]
            print('kf: ', k)
            print('epoch_num: ', epoch_num + 1)

            model.fit_generator(
                train_batch_generator(x_train, y_train, config.batch_size),
                epochs=5,
                steps_per_epoch=int(y_train.shape[0] / config.batch_size),
                validation_data=(x_dev, y_dev),
            )
            pred = model.predict(x_dev, batch_size=config.batch_size)
            pre, rec, f1 = score(y_dev, pred)
            y_t=np.argmax(y_dev, axis=1)+1
            y_p = np.argmax(pred, axis=1)+1
            print(y_t)
            print(y_p)
            acc = accuracy_score(y_t, y_p)

            S_train[test_index, epoch_num] = pred[:, 1]
            print('acc:',acc)
            train_df['epoch_{0}'.format(epoch_num)] = S_train[:, epoch_num]
            train_df['label'] = Y_train[:, 1]
            a.append(acc)
        model.save(config.stack_path+"_%s_%s.h5" %
                   (model_name, k))

    print(np.array(a).T)
    print('mean :', np.mean(np.array(a)))
    train_df.to_csv(config.stack_path+'train_%s.csv' % (k),
                    index=False, )


def do_train_cv(model_name, model, epoch_nums, kfolds,lr):
    X_train, Y_train = load_data()
    make_train_cv_data(X_train, Y_train, model, model_name, epoch_nums, kfolds,lr)


def main(model_name):
    print('model name', model_name)
    path = config.origin_csv
    from w2v import load_my_train_w2v,load_pre_train_w2v
    vocab, embed_weights = load_my_train_w2v(config.origin_csv)
    lr = 0.01
    if model_name == 'cnn1':
        model = cnn_v1(config.word_maxlen,
                       embed_weights, pretrain=True)
    #0.58
    if model_name == 'cnn2':
        model = cnn_v2(config.word_maxlen,
                       embed_weights, pretrain=True)
    #0.62
    if model_name == 'rnn1':
        model = rnn_v1(config.word_maxlen,
                      embed_weights, pretrain=True)
    
    #0.61
    if model_name == 'rcnn1':
        model = rcnn_v1(config.word_maxlen,
                       embed_weights, pretrain=True,trainable=True)
    if model_name == 'rnn_att':
        model = rnn_att()
    if model_name == 'rnn_att2':
        model = rnn_att2()
    do_train_cv(model_name, model, epoch_nums=1, kfolds=5,lr=lr)
    #train(x_train, y_train, x_dev, y_dev, model_name, model)

if __name__ == '__main__':

    main(sys.argv[1])
    # do_cv()
