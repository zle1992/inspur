#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)
import sys
import keras
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import numpy as np
from keras.activations import softmax
from keras import backend
import pandas as pd
# Model Load
sys.path.append('utils/')
import config
from help import get_X_Y_from_df

from CutWord import cut_word,read_cut
from Data2id import data2id


sys.path.append('models')
from CNN import cnn_v1,cnn_v2,rnn_v1,rcnn_v1
from RNN import rnn_att,rnn_att2
sys.path.append('utils/')
import config
from base import TextModel, Attention
MAX_LEN = config.word_maxlen 

from train import get_model

CustomObjects={
"softmax": softmax,
'Attention':Attention
}

def make_test_cv_data(data, model_name, kfolds):

    X_dev, _ = get_X_Y_from_df(data)
    test_df = pd.DataFrame()
    pred_probs = np.zeros((X_dev.shape[0],3))
    
    for kf in range(1, kfolds + 1):
        print('kf: ', kf)
        bst_model_path = config.stack_path + \
        "dp_%s_%d.h5" % (model_name,kf)
        model,lr = get_model(model_name)
        model.load_weights(bst_model_path)
        model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),metrics = ['acc'])
        pred = model.predict(X_dev, batch_size=config.batch_size)
       
        pred_probs +=pred
    pred_probs /=kfolds
    print(pred_probs.shape)
    test_df['id'] = data.id.values
    test_df['{0}_0'.format(model_name)] = pred_probs[:,0]
    test_df['{0}_1'.format(model_name)] = pred_probs[:,1]
    test_df['{0}_2'.format(model_name)] = pred_probs[:,2]
    test_df['label'] =  np.argmax(pred_probs, axis=1) +1
    test_df.to_csv(config.stack_path+'test_%s.csv' % (model_name),
                   index=False,)
    return pred_probs

    




def single_submit(data,model_name):
    X, _ = get_X_Y_from_df(data)
    bst_model_path = config.model_dir + \
        "dp_embed_%s.h5" % ( model_name)
    model,lr = get_model(model_name)
    model.load_weights(bst_model_path)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),metrics = ['acc'])
    test_pred = model.predict(X, batch_size=config.batch_size)
    return test_pred



def submit(in_path,out_path,model_name,cv=False):
    data = read_cut(in_path,config.test_data_cut_hdf)
    data = data2id(data)
    data.label = data.label.fillna(0)
    
    print('load model and predict')
    if not cv:
        test_pred = single_submit(data,model_name)
    else:
        test_pred = make_test_cv_data(data, model_name, kfolds=10)
    test_model_pred = np.squeeze(test_pred)
    data['label'] = np.argmax(test_model_pred, axis=1) +1
    data[['label']].to_csv(
        out_path, index=False, header=None, sep='\t')
if __name__ == '__main__':

    cv = False
    cv = True
    model_name = 'cnn1'
    in_path = 'submit/Preliminary-texting.csv'
    out_path = 'submit/dsjyycxds_preliminary.txt'
    submit(in_path, out_path,model_name,cv)

    
