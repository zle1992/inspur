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
sys.path.append('utils/')
import config


def convs_block(data, convs=[3, 4, 5], f=256):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(
            Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools)


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


def cnn_v1(seq_length, embed_weight, pretrain=False):

    main_input = Input(shape=(seq_length,), dtype="int32")

    in_dim, out_dim = embed_weight.shape
    embedding = Embedding(input_dim=in_dim, weights=[
        embed_weight], output_dim=out_dim, trainable=True)

    q1_q2 = Activation(activation="relu")(
        BatchNormalization()((TimeDistributed(Dense(256))(embedding(main_input)))))

    q1_q2 = convs_block(q1_q2)
    q1_q2 = Dropout(0.5)(q1_q2)
    fc = Activation(activation="relu")(
        BatchNormalization()(Dense(256)(q1_q2)))
    output = Dense(3, activation="softmax")(fc)
    print(output)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model


def cnn_v2(seq_length, embed_weight, pretrain=False):
    '''
    deep cnn conv + maxpooling + conv + maxpooling
    '''
    content = Input(shape=(seq_length,), dtype="int32")
    in_dim, out_dim = embed_weight.shape
    embedding = Embedding(input_dim=in_dim, weights=[
        embed_weight], output_dim=out_dim, trainable=False)

    trans_content = Activation(activation="relu")(
        BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
    feat = convs_block2(trans_content, convs=[1, 2, 3, 4, 5, 6, 7])

    dropfeat = Dropout(0.5)(feat)
    fc = Activation(activation="relu")(
        BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(3, activation="softmax")(fc)
    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model




def rnn_v1(seq_length, embed_weight, pretrain=False):
    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接

    main_input = Input(shape=(seq_length,), dtype='float64')

    # 词嵌入（使用预训练的词向量）

    in_dim, out_dim = embed_weight.shape
    embedding = Embedding(input_dim=in_dim, weights=[
        embed_weight], output_dim=out_dim, trainable=False)
    content = Activation(activation="relu")(
        BatchNormalization()((TimeDistributed(Dense(256))(embedding(main_input)))))
    content = Bidirectional(CuDNNGRU(256))(content)
    content = Dropout(0.5)(content)
    fc = Activation(activation="relu")(
        BatchNormalization()(Dense(256)(content)))
    main_output = Dense(3,
                        activation='softmax')(fc)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def rcnn_v1(seq_length, embed_weight, pretrain=False,trainable=False):
    # 模型结构：词嵌入-卷积池化

    main_input = Input(shape=(seq_length,), dtype='float64')

    # 词嵌入（使用预训练的词向量）

    in_dim, out_dim = embed_weight.shape
    embedding = Embedding(input_dim=in_dim, weights=[
        embed_weight], output_dim=out_dim, trainable=False)

    content = embedding(main_input)
    trans_content = Activation(activation="relu")(
        BatchNormalization()((TimeDistributed(Dense(256))(content))))
    conv = Activation(activation="relu")(BatchNormalization()(
        Conv1D(filters=128, kernel_size=5, padding="valid")(trans_content)))

    cnn1 = conv
    cnn1 = MaxPool1D(pool_size=5)(cnn1)
    gru = Bidirectional(CuDNNGRU(128))(cnn1)

    merged = Activation(activation="relu")(gru)
    merged = Dropout(0.2)(merged)
    main_output = Dense(3,
                        activation='softmax')(merged)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


