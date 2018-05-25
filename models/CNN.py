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


def convs_block(data, convs=[3, 3, 4, 5, 5, 7, 7], f=256):
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
        conv = MaxPool1D(pool_size=10)(conv)
        conv = Activation(activation="relu")(BatchNormalization()(
            Conv1D(filters=f, kernel_size=c, padding="valid")(conv)))

        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)


def cnn_v1(seq_length, embed_weight, pretrain=False):

    q1_q2 = Input(shape=(seq_length,), dtype="int32")

    in_dim, out_dim = embed_weight.shape
    embedding = Embedding(input_dim=in_dim, weights=[
        embed_weight], output_dim=out_dim, trainable=False)

    q1_q2 = Activation(activation="relu")(
        BatchNormalization()((TimeDistributed(Dense(256))(embedding(q1_q2)))))

    q1_q2 = convs_block(q1_q2)

    q1_q2 = Dropout(0.5)(q1_q2)
    fc = Activation(activation="relu")(
        BatchNormalization()(Dense(256)(q1_q2)))
    output = Dense(3, activation="softmax")(fc)
    print(output)
    model = Model(inputs=q1_input, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model


# def get_textcnn3(seq_length, embed_weight, pretrain=False):
#     '''
#     deep cnn conv + maxpooling + conv + maxpooling
#     '''
#     content = Input(shape=(seq_length,), dtype="int32")
#     if pretrain:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#                               'vocab_size'], weights=[embed_weight], output_dim=config['w2v_vec_dim'], trainable=False)
#     else:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#                               'vocab_size'], output_dim=config['w2v_vec_dim'], trainable=True)
#     trans_content = Activation(activation="relu")(
#         BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
#     feat = convs_block2(trans_content)

#     dropfeat = Dropout(0.2)(feat)
#     fc = Activation(activation="relu")(
#         BatchNormalization()(Dense(256)(dropfeat)))
#     output = Dense(2, activation="softmax")(fc)
#     model = Model(inputs=content, outputs=output)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer="adam", metrics=['accuracy'])
#     model.summary()
#     return model


# def get_textcnn2(seq_length, embed_weight, pretrain=False):
#     # 模型结构：词嵌入-卷积池化-卷积池化-flat-drop-softmax

#     main_input = Input(shape=(seq_length,), dtype='float64')

#     # 词嵌入（使用预训练的词向量）

#     if pretrain:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#                               'vocab_size'], weights=[embed_weight], output_dim=config['w2v_vec_dim'], trainable=False)
#     else:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#                               'vocab_size'], output_dim=config['w2v_vec_dim'], trainable=True)
#     embed = embedding(main_input)

#     cnn = Activation(activation='relu')(BatchNormalization()(
#         Convolution1D(filters=256, kernel_size=3, padding='valid')(embed)))
#     cnn = MaxPool1D(pool_size=4)(cnn)

#     cnn = Activation(activation='relu')(BatchNormalization()(
#         Convolution1D(filters=256, kernel_size=3, padding='valid')(cnn)))
#     #cnn = MaxPool1D(pool_size=4)(cnn)
#     cnn = GlobalMaxPool1D()(cnn)
#     #cnn = Flatten()(cnn)
#     drop = Dropout(0.2)(cnn)
#     main_output = Dense(config['number_classes'], activation='softmax')(drop)
#     model = Model(inputs=main_input, outputs=main_output)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=Adam(), metrics=['accuracy'])
#     return model


# def get_textrnn(seq_length, embed_weight, pretrain=False):
#     # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接

#     main_input = Input(shape=(seq_length,), dtype='float64')

#     # 词嵌入（使用预训练的词向量）

#     if pretrain:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#             'vocab_size'], weights=[embed_weight], output_dim=config['w2v_vec_dim'], trainable=False)
#     else:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#             'vocab_size'], output_dim=config['w2v_vec_dim'], trainable=True)
#     content = embedding(main_input)
#     # trans_content = Activation(activation="relu")(
#     #     BatchNormalization()((TimeDistributed(Dense(256))(embedding(co
#     # print('Build model...')
#     embed = Bidirectional(GRU(256))(content)

#     # merged = layers.add([encoded_sentence, encoded_question])
#     merged = BatchNormalization(embed)
#     merged = Dropout(0.3)(merged)
#     fc = Activation(activation="relu")(
#         BatchNormalization()(Dense(256)(merged)))
#     main_output = Dense(config['number_classes'],
#                         activation='softmax')(fc)

#     model = Model(inputs=main_input, outputs=main_output)
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model


# def get_textrcnn(seq_length, embed_weight, pretrain=False,trainable=False):
#     # 模型结构：词嵌入-卷积池化

#     main_input = Input(shape=(seq_length,), dtype='float64')

#     # 词嵌入（使用预训练的词向量）

#     if pretrain:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#             'vocab_size'], weights=[embed_weight], output_dim=config['w2v_vec_dim'], trainable=trainable)
#     else:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#             'vocab_size'], output_dim=config['w2v_vec_dim'], trainable=True)

#     print('Build model...')
#     content = embedding(main_input)
#     trans_content = Activation(activation="relu")(
#         BatchNormalization()((TimeDistributed(Dense(256))(content))))
#     conv = Activation(activation="relu")(BatchNormalization()(
#         Conv1D(filters=128, kernel_size=5, padding="valid")(trans_content)))

#     cnn1 = conv
#     cnn1 = MaxPool1D(pool_size=5)(cnn1)
#     gru = Bidirectional(GRU(128))(cnn1)

#     merged = Activation(activation="relu")(gru)
#     merged = Dropout(0.2)(merged)
#     main_output = Dense(config['number_classes'],
#                         activation='softmax')(merged)

#     model = Model(inputs=main_input, outputs=main_output)
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model


# def model3():
#     sequence_length = x_text.shape[1]  # 56
#     vocabulary_size = config['vocab_size'] + 1  # 18765
#     embedding_dim = 256
#     filter_sizes = [3, 4, 5]
#     num_filters = 512
#     drop = 0.5

#     epochs = 100
#     batch_size = 30

#     # this returns a tensor
#     print("Creating Model...")
#     inputs = Input(shape=(sequence_length,), dtype='int32')
#     embedding = Embedding(input_dim=vocabulary_size + 1, weights=[
#         embed_weight], output_dim=embedding_dim, trainable=True, input_length=config['word_maxlen'])(inputs)
#     # embedding = Embedding(input_dim=vocabulary_size,
#     # output_dim=embedding_dim, input_length=sequence_length)(inputs)

#     print('reshape')
#     reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

#     conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[
#         0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
#     conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[
#         1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
#     conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[
#         2], embedding_dim), padding='valid', kernel_initializer='normal',
#         activation='relu')(reshape)

#     maxpool_0 = MaxPool2D(pool_size=(
#         sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
#     maxpool_1 = MaxPool2D(pool_size=(
#         sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
#     maxpool_2 = MaxPool2D(pool_size=(
#         sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1),
#         padding='valid')(conv_2)

#     concatenated_tensor = Concatenate(axis=1)(
#         [maxpool_0, maxpool_1, maxpool_2])
#     flatten = Flatten()(concatenated_tensor)
#     dropout = Dropout(drop)(flatten)
#     output = Dense(units=2, activation='softmax')(dropout)

#     # this creates a model that includes
#     model = Model(inputs=inputs, outputs=output)

#     checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5',
#                                  monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
#     adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#     model.compile(optimizer=adam, loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     print("Traning Model...")
#     model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
#               callbacks=[checkpoint], validation_data=(x_dev, y_dev))  # starts
#     training
