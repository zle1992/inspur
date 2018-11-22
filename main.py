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
from CNN import TextCNN ,TextCNN2
from RNN import TextRNN,TextRNN2
sys.path.append('utils/')
import config

from help import score, train_batch_generator, train_test, get_X_Y_from_df
from CutWord import cut_word,read_cut
from Data2id import data2id
from sklearn.metrics import accuracy_score,confusion_matrix






   
def get_model(model_name):
    

    
    if model_name == 'cnn1':
        model = TextCNN(
                model_name ='CNN',
                nb_epoch=10, 
                max_len=config.word_maxlen, 
                embed_size=config.embed_size,
                batch_size=128,
                lr=0.01,
                kfold =5,
                word_embed_weight=config.word_embed_weight,
                stack_path = config.stack_path,
                model_dir = config.model_dir,
                use_pretrained=True,
                trainable=True,
               # **kwargs
                )
    if model_name == 'cnn2':
        model = TextCNN(
                model_name ='CNN2',
                nb_epoch=10, 
                max_len=config.word_maxlen, 
                embed_size=config.embed_size,
                batch_size=128,
                lr=0.01,
                kfold =5,
                word_embed_weight=config.word_embed_weight,
                stack_path = config.stack_path,
                use_pretrained=True,
                trainable=True,
               # **kwargs
                )

    if model_name == 'rnn1':
        model = TextRNN(
                model_name ='RNN',
                nb_epoch=10, 
                max_len=config.word_maxlen, 
                embed_size=config.embed_size,
                batch_size=128,
                lr=0.01,
                kfold =5,
                word_embed_weight=config.word_embed_weight,
                stack_path = config.stack_path,
                model_dir = config.model_dir,
                use_pretrained=True,
                trainable=True,
               # **kwargs
                )
    #0.66
    if model_name == 'rnn2':
        model = TextRNN2(
                model_name ='RNN2',
                nb_epoch=10, 
                max_len=config.word_maxlen, 
                embed_size=config.embed_size,
                batch_size=128,
                lr=0.01,
                kfold =5,
                word_embed_weight=config.word_embed_weight,
                stack_path = config.stack_path,
                model_dir = config.model_dir,
                use_pretrained=True,
                trainable=True,
               # **kwargs
                )
    return model





def train(cv,model_name):

    data_df = read_cut(config.origin_csv,config.train_data_cut_hdf)
    data_df = data2id(data_df)
    data_df = data_df.sample(frac=1, random_state=18)

    model = get_model(model_name)

    if cv:
        kfolds=5
        x_train, y_train = get_X_Y_from_df(data_df)
        model.make_train_cv_data([x_train, y_train],kfolds)
    else:
        train, dev = train_test(data_df)
        x_train, y_train = get_X_Y_from_df(train)
        x_dev, y_dev = get_X_Y_from_df(dev)
        model.single_train([x_train, y_train ,x_dev, y_dev ])



def submit_inteface(in_path,out_path,model_name,cv=False):


    data_df = read_cut(in_path,config.test_data_cut_hdf)
    data_df = data2id(data_df)
    data_df.label = data_df.label.fillna(0)
    X, _ = get_X_Y_from_df(data_df)
    data = [X,_]
    model = get_model(model_name)


    print('load model and predict')
    if not cv:
        test_pred = model.single_predict(data)
    else:
        test_pred = model.make_test_cv_data(data)
    test_model_pred = np.squeeze(test_pred)
    data_df['label'] = np.argmax(test_model_pred, axis=1) +1
    data_df[['label']].to_csv(
        out_path, index=False, header=None, sep='\t')



def submit(cv,model_name):
    in_path = 'submit/Preliminary-texting.csv'
    out_path = 'submit/dsjyycxds_preliminary2.txt'
    submit_inteface(in_path, out_path,model_name,cv)

if __name__ == '__main__':

    m,model_name=sys.argv[1],sys.argv[2]

    #train(m=='cv',model_name)
    submit(m=='cv',model_name)
