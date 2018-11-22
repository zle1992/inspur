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
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from keras import backend as K
from BaseModel import TextModel





class TextCNN(TextModel):
    """docstring for TextCNN"""
    def __init__(self,**kwargs):
        #TextModel.__init__(self,**kwargs)
        super(TextCNN, self).__init__(**kwargs)
        self.filters = [3, 4, 5]
        self.number_classes =3
        
    


    def convs_block(self,data, convs=[3, 4, 5], f=256):
        pools = []
        for c in convs:
            conv = Activation(activation="relu")(BatchNormalization()(
                Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
            pool = GlobalMaxPool1D()(conv)
            pools.append(pool)
        return concatenate(pools)




    def get_model(self, trainable=None):


        main_input = Input(shape=(self.max_len,), dtype="int32")

        embedding = self.create_embedding(mask_zero=False)

        # review = Activation(activation="relu")(
        #     BatchNormalization()((TimeDistributed(Dense(256))(embedding(main_input)))))


        review = embedding(main_input)
        review = self.convs_block(review)
        review = Dropout(0.2)(review)
        fc = Activation(activation="relu")(
            BatchNormalization()(Dense(256)(review)))

        output = Dense(self.number_classes, activation="softmax")(fc)
       
        return main_input,output

    def _get_bst_model_path(self):
        return self.model_dir+"/{pre}_{epo}_{embed}_{max_len}_{wind}_{time}.h5".format(
            pre = self.model_name,
            epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
            time="now")#self.time,)# upt=int(self.use_pretrained), tn=int(self.trainable))


        # return "{pre}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}.h5".format(
        #     pre=self.__class__.__name__, epo=self.nb_epoch,
        #     embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
        #     time=self.time,)# upt=int(self.use_pretrained), tn=int(self.trainable))
        


class TextCNN2(TextCNN):
    """docstring for TextCNN"""
    def __init__(self,):
        self.filters = [3, 4, 5]
        self.model_name = 'CNN2'
        super(TextCNN2, self).__init__()
    


    def convs_block2(data, convs=[3, 4, 5], f=256, name="conv_feat"):
        pools = []
        for c in convs:
            conv = Activation(activation="relu")(BatchNormalization()(
                Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
            #conv = MaxPool1D(pool_size=10)(conv)
            conv = Activation(activation="relu")(BatchNormalization()(
                Conv1D(filters=f, kernel_size=c, padding="valid")(conv)))

            pool = GlobalMaxPool1D()(conv)
            pools.append(pool)
        return concatenate(pools, name=name)




    def get_model(self, trainable=None):

 
        '''
        deep cnn conv + maxpooling + conv + maxpooling
        '''
        main_input = Input(shape=(self.word_maxlen,), dtype="int32")

        embedding = create_pretrained_embedding(mask_zero=False)

        trans_content = Activation(activation="relu")(
            BatchNormalization()((TimeDistributed(Dense(256))(embedding(main_input)))))
        feat = convs_block2(trans_content, convs=[1, 2, 3,])

        dropfeat = Dropout(0.5)(feat)
        fc = Activation(activation="relu")(
            BatchNormalization()(Dense(256)(dropfeat)))
        output = Dense(self.number_classes, activation="softmax")(fc)
       
        return main_input,output














# def rnn_v1():
    

#     main_input = Input(shape=(config.word_maxlen,), dtype="int32")

#     embedding = create_pretrained_embedding(mask_zero=False)

#     content = Activation(activation="relu")(
#         BatchNormalization()((TimeDistributed(Dense(256))(embedding(main_input)))))
#     content = Bidirectional(CuDNNGRU(256))(content)
#     content = Dropout(0.5)(content)
#     fc = Activation(activation="relu")(
#         BatchNormalization()(Dense(256)(content)))
#     main_output = Dense(3,
#                         activation='softmax')(fc)

#     model = Model(inputs=main_input, outputs=main_output)
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     model.summary()
#     return model


# def rcnn_v1():

#     main_input = Input(shape=(config.word_maxlen,), dtype="int32")

#     embedding = create_pretrained_embedding(mask_zero=False)

#     content = embedding(main_input)
#     trans_content = Activation(activation="relu")(
#         BatchNormalization()((TimeDistributed(Dense(64))(content))))
#     conv = Activation(activation="relu")(BatchNormalization()(
#         Conv1D(filters=32, kernel_size=3, padding="valid")(trans_content)))

#     cnn1 = conv
#     cnn1 = MaxPool1D(pool_size=5)(cnn1)
#     gru = Bidirectional(CuDNNGRU(128))(cnn1)

#     merged = Activation(activation="relu")(gru)
#     merged = Dropout(0.2)(merged)
#     main_output = Dense(3,
#                         activation='softmax')(merged)

#     model = Model(inputs=main_input, outputs=main_output)
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     model.summary()
#     return model


