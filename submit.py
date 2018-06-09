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
from process import read_data, make_w2v
from help import get_X_Y_from_df





def main(model_path):
    in_path = 'submit/Preliminary-texting.csv'
    out_path = 'submit/{0}_dsjyycxds_preliminary.txt'.format(model_path.split('/')[-1])
    vocab, embed_weights = make_w2v(config.origin_csv)
    data = read_data(in_path)
    data.label = data.label.fillna(0)
    X, _ = get_X_Y_from_df(data)
    print('load model and predict')
    model = load_model(model_path, custom_objects={"softmax": softmax})

    test_model_pred = np.squeeze(model.predict(X))
    data['label'] = np.argmax(test_model_pred, axis=1) +1
    data[['label']].to_csv(
        out_path, index=False, header=None, sep='\t')

if __name__ == '__main__':

    main(sys.argv[1])
    
