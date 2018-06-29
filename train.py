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
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard, ReduceLROnPlateau

sys.path.append('models')
from CNN import cnn_v1,cnn_v2,rnn_v1,rcnn_v1
from RNN import rnn_att,rnn_att2
sys.path.append('utils/')
import config

from help import score, train_batch_generator, train_test, get_X_Y_from_df
from CutWord import cut_word,read_cut
from Data2id import data2id
from sklearn.metrics import accuracy_score,confusion_matrix




def train_model(x_train, y_train, x_dev, y_dev, model, lr,bst_model_path):
   
    model_checkpoint = ModelCheckpoint(
        bst_model_path, monitor='val_acc', save_best_only=True, save_weights_only=True,mode = 'max')
    early_stopping =  EarlyStopping(monitor='val_acc', patience=4,
                    mode='max')
    change_lr=ReduceLROnPlateau(monitor='val_acc',mode = 'max',factor=0.1,epsilon=0.001,min_lr=0.0001,patience=1)
    
    K.set_value(model.optimizer.lr, lr)
    model.fit(x_train, y_train,
        epochs=10,
        validation_data=(x_dev, y_dev),
        batch_size=config.batch_size,
        callbacks=[model_checkpoint,early_stopping,change_lr,
        # TensorBoard(log_dir='data/log_dir'),
         ],
    )

    model.load_weights(bst_model_path)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),metrics = ['acc'])
    return model

def make_train_cv_data(data, model_name, kfolds):
    X_train, Y_train = get_X_Y_from_df(data)
    S_train = np.zeros((Y_train.shape[0], 5))
    train_df= pd.DataFrame()
    X, Y = X_train, Y_train
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=kfolds, shuffle=True,random_state=2018)
    k = 0
    a= []    
    for train_index, test_index in kf.split(Y):
        k += 1
        bst_model_path = config.stack_path + \
        "dp_%s_%d.h5" % (model_name,k)
        
        model,lr = get_model(model_name)

        x_train = X_train[train_index]
        x_dev = X_train[test_index]
        y_train=Y_train[train_index]
        y_dev = Y_train[test_index]
        id_dev =  data.id.values[test_index]
        print('kf: ', k)

        model_trained = train_model(x_train, y_train, x_dev, y_dev, model, lr,bst_model_path)
        pred = model_trained.predict(x_dev)
        y_t=np.argmax(y_dev, axis=1)
        y_p = np.argmax(pred, axis=1)
        acc = accuracy_score(y_t, y_p)
        print(confusion_matrix(y_t, y_p))
        S_train[test_index, :3] = pred
        S_train[test_index,3] = id_dev
        S_train[test_index,4] = y_t
        print('acc:',acc)
        a.append(acc)

    train_df['id'] = S_train[:,3]
    train_df['{0}_0'.format(model_name)] = S_train[:,0]
    train_df['{0}_1'.format(model_name)] = S_train[:,1]
    train_df['{0}_2'.format(model_name)] = S_train[:,2]
    train_df['label'] = S_train[:,4]

    print('acc ',a)
        
    print('mean :',np.mean(np.array(a)))
    train_df.to_csv(config.stack_path+'train_%s.csv' % (k),
                    index=False, )



   
def get_model(model_name):
    lr = 0.001

    #0.65
    if model_name == 'cnn1':
        model = cnn_v1()

    #0.
    if model_name == 'cnn2':
        model = cnn_v2()
    #0.
    if model_name == 'rnn1':
        model = rnn_v1()
    
    #0.66
    if model_name == 'rcnn1':
        model = rcnn_v1()
    #0.65
    if model_name == 'rnn_att':
        model = rnn_att()
    #0.66
    if model_name == 'rnn_att2':
        model = rnn_att2()
    return model,lr


def cv(model_name):
    kfolds=10
    data = read_cut(config.origin_csv,config.train_data_cut_hdf)
    data = data2id(data)
    data = data.sample(frac=1, random_state=18)
    make_train_cv_data(data, model_name, kfolds)
    
def single_train(model_name):
    print('model name', model_name)
    bst_model_path = config.model_dir + \
        "dp_embed_%s.h5" % ( model_name)

    data = read_cut(config.origin_csv,config.train_data_cut_hdf)
    data = data2id(data)
    data = data.sample(frac=1, random_state=18)
    train, dev = train_test(data)
    x_train, y_train = get_X_Y_from_df(train)
    x_dev, y_dev = get_X_Y_from_df(dev)

    model,lr = get_model(model_name)

    model = train_model(x_train, y_train, x_dev, y_dev, model, lr,bst_model_path)

    pred = model.predict(x_dev)
    y_t=np.argmax(y_dev, axis=1)
    y_p = np.argmax(pred, axis=1)
    acc = accuracy_score(y_t, y_p)
    print('acc: ' ,acc)

if __name__ == '__main__':

    m,model_name=sys.argv[1],sys.argv[2]
    if m=='cv':
        cv(model_name)
    else:
        single_train(model_name)



