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
                          pretrained_weights], trainable=True, **kwargs)
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


def rnn_att(pretrained_embedding=config.word_embed_weight,
         maxlen=MAX_LEN,
         lstm_dim=300,
         dense_dim=300,
         dense_dropout=0.5):

    # Based on arXiv:1609.06038
    inputs = Input(name='q1', shape=(maxlen,))
    

    # Embedding
    embedding = create_pretrained_embedding(
        pretrained_embedding, mask_zero=False)
    bn = BatchNormalization()
    emb = bn(embedding(inputs))

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(emb)
    x = attention_3d_block(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(3, activation='linear')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model

def rnn_att2(pretrained_embedding=config.word_embed_weight, LSTM_hidden_size = 128):

    # Based on arXiv:1609.06038
    inputs = Input(name='inputs', shape=(MAX_LEN,))
    

    # Embedding
    embedding = create_pretrained_embedding(
        pretrained_embedding, mask_zero=False)
    bn = BatchNormalization()
    emb = bn(embedding(inputs))
    emb = SpatialDropout1D(0.5)(emb)
    x = Bidirectional(CuDNNGRU(LSTM_hidden_size, return_sequences=True))(emb)
    x = Bidirectional(CuDNNGRU(LSTM_hidden_size, return_sequences=True))(x)
    x = Attention(MAX_LEN)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(3, activation='linear')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='mse',
                  optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model