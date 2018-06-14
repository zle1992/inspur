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
import pandas as pd

sys.path.append('utils/')
sys.path.append('models/')
import config
from base import TextModel, Attention
MAX_LEN = config.word_maxlen 






def make_test_cv_data(X_dev, model_name, epoch_nums, kfolds):
    mean_epoch = False
    test_df = pd.DataFrame()
    S_test = np.zeros((X_dev.shape[0], epoch_nums))
    pred_probs = np.zeros((X_dev.shape[0],3))
    for epoch_num in range(epoch_nums):
        for kf in range(1, kfolds + 1):
            print('kf: ', kf)
            print('epoch_num: ', epoch_num + 1)
            model = load_model(config.stack_path+"_%s_%s.h5" %
                               (model_name, kf), custom_objects={"softmax": softmax})
            pred = model.predict(X_dev, batch_size=config.batch_size)
            pred_probs +=pred
        
    return pred_probs
        # test_df['epoch_%s' % (epoch_num)] = S_test[:, epoch_num]
        # test_df.to_csv(config.stack_path+'test_%s.csv' % (model_name),
        #                index=False,)
        # if mean_epoch:
        #     pred = np.mean(S_test, axis=1)
        # else:
        #     pred = S_test[:,epoch_num]
        # return pred


def do_cv_test():

    model_name = 'cnn1'
    epoch_nums = 1
    kfolds = 5

    in_path = 'submit/Preliminary-texting.csv'
    out_path = 'submit/{0}_dsjyycxds_preliminary.txt'.format(model_name)
    data = read_cut(in_path,config.test_data_cut_hdf)
    data = data2id(data)
    data.label = data.label.fillna(0)
    X, _ = get_X_Y_from_df(data)
    pred = make_test_cv_data(X, model_name, epoch_nums, kfolds)
    data['label'] = np.argmax(pred, axis=1) +1
    data[['label']].to_csv(
        out_path, index=False, header=None, sep='\t')

if __name__ == '__main__':
    #main(sys.argv[1], sys.argv[2])
    do_cv_test()
    # main_test(sys.argv[1])
