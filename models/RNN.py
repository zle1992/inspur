import numpy as np
import pandas as pd
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K
import sys
sys.path.append('utils/')
import config
from base import TextModel, Attention
MAX_LEN = config.word_maxlen 


def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[
                          pretrained_weights], trainable=trainable, **kwargs)
    return embedding


def attention_3d_block(inputs):
    """
    attention mechanisms for lstm
    :param inputs: shape (batch_size, seq_length, input_dim)
    :return:
    """
    a = Permute((2, 1))(inputs)
    a = Dense(MAX_LEN, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)    # attention_vec
    att_mul = multiply([inputs, a_probs])
    return att_mul


def rnn_att(
         lstm_dim=64,
         dense_dim=128,
         dense_dropout=0.5):

    # Based on arXiv:1609.06038
    inputs = Input(name='inputs', shape=(config.word_maxlen,))
    

    # Embedding
    embedding = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)
    bn = BatchNormalization()
    emb = bn(embedding(inputs))

    x = Bidirectional(CuDNNGRU(lstm_dim, return_sequences=True))(emb)
    x = attention_3d_block(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(dense_dim, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model

def rnn_att2( LSTM_hidden_size = 128):

    # Based on arXiv:1609.06038
    inputs = Input(name='inputs', shape=(config.word_maxlen,))
    

    # Embedding
    embedding = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)
    bn = BatchNormalization()
    #emb = bn(embedding(inputs))
    emb = embedding(inputs)
    emb = SpatialDropout1D(0.2)(emb)

    #emb =embedding(inputs)
    x = Bidirectional(CuDNNLSTM(LSTM_hidden_size, return_sequences=True))(emb)
    x = Bidirectional(CuDNNLSTM(LSTM_hidden_size, return_sequences=True))(x)
    x = Attention(config.word_maxlen)(x)
    x = Dropout(0.2)(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model