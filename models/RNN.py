import numpy as np
import pandas as pd
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K
import sys
from base import Attention
from BaseModel import TextModel

class TextRNN(TextModel):
    """docstring for TextRNN"""
    def __init__(self, **kwargs):
        super(TextRNN, self).__init__(**kwargs)
        self.number_classes =3
        self.lstm_dim =128
        self.dense_dim = 256

    def attention_3d_block(self,inputs):
        """
        attention mechanisms for lstm
        :param inputs: shape (batch_size, seq_length, input_dim)
        :return:
        """
        a = Permute((2, 1))(inputs)
        a = Dense(self.max_len, activation='softmax')(a)
        a_probs = Permute((2, 1))(a)    # attention_vec
        att_mul = multiply([inputs, a_probs])
        return att_mul
    def get_model(self, trainable=None):


        main_input = Input(shape=(self.max_len,), dtype="int32")

        embedding = self.create_embedding(mask_zero=False)

        bn = BatchNormalization()
        emb = bn(embedding(main_input))

        x = Bidirectional(CuDNNGRU(self.lstm_dim, return_sequences=True))(emb)
        x = self.attention_3d_block(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(self.dense_dim, activation='relu')(x)
        output = Dense(self.number_classes, activation='softmax')(x)
    
       
        return main_input,output

    def _get_bst_model_path(self):
        return self.model_dir+"/{pre}_{epo}_{embed}_{max_len}_{lstm_dim}_{time}.h5".format(
            pre = self.model_name,
            epo=self.nb_epoch,
            embed=self.embed_size, 
            max_len=self.max_len, 
            lstm_dim=self.lstm_dim,
            time="now")#self.time,)# upt=int(self.use_pretrained), tn=int(self.trainable))



class TextRNN2(TextRNN):
    """docstring for TextRNN"""
    def __init__(self, **kwargs):
        super(TextRNN2, self).__init__(**kwargs)
        self.lstm_dim =128
        self.dense_dim = 256
    def get_model(self, trainable=None):


        main_input = Input(shape=(self.max_len,), dtype="int32")

        embedding = self.create_embedding(mask_zero=False)


        bn = BatchNormalization()
        #emb = bn(embedding(inputs))
        emb = embedding(main_input)
        emb = SpatialDropout1D(0.2)(emb)

        #emb =embedding(inputs)
        x = Bidirectional(CuDNNLSTM(self.lstm_dim, return_sequences=True))(emb)
        x = Bidirectional(CuDNNLSTM(self.lstm_dim, return_sequences=True))(x)
        x = Attention(self.max_len)(x)
        x = Dropout(0.2)(x)
        x = Dense(self.dense_dim, activation='relu')(x)
        output = Dense(self.number_classes, activation='softmax')(x)

       
        return main_input,output

    