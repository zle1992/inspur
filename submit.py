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
# Model Load
sys.path.append('utils/')
import config
from help import get_X_Y_from_df

from CutWord import cut_word,read_cut
from Data2id import data2id


sys.path.append('utils/')
sys.path.append('models/')
import config
from base import TextModel, Attention
MAX_LEN = config.word_maxlen 

def train(model_name, model):
    print('load data')
    data = read_cut(config.origin_csv)
    data = data2id(data)
    train, dev = train_test(data)
    x_train, y_train = get_X_Y_from_df(train)
    x_dev, y_dev = get_X_Y_from_df(dev)



def main(model_path):
    in_path = 'submit/Preliminary-texting.csv'
    out_path = 'submit/{0}_dsjyycxds_preliminary.txt'.format(model_path.split('/')[-1])
    data = read_cut(in_path,config.test_data_cut_hdf)
    data = data2id(data)
    data.label = data.label.fillna(0)
    X, _ = get_X_Y_from_df(data)
    print('load model and predict')
    model = load_model(model_path, custom_objects={"softmax": softmax,"Attention":Attention})

    test_model_pred = np.squeeze(model.predict(X))
    data['label'] = np.argmax(test_model_pred, axis=1) +1
    data[['label']].to_csv(
        out_path, index=False, header=None, sep='\t')

if __name__ == '__main__':

    main(sys.argv[1])
    
