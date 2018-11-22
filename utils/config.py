#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
# sys.path.append('models/')

# from base import Attention
# from keras.activations import softmax
# CustomObjects={
# "softmax": softmax,
# 'Attention':Attention
# }

batch_size = 128
number_classes = 3
w2v_vec_dim = 256
embed_size =  256
word_maxlen = 60
cut_char_level = False

model_dir = 'data/model_dir/'
jieba_dict = 'data/jieba/jieba_dict.txt'
stopwords_path = 'data/jieba/stops.txt'
test_csv = 'submit/Preliminary-texting.csv'
stack_path = 'data/stack/'
origin_csv = 'data/training-inspur.csv'

#w2v
word_embed_weight = 'data/my_w2v/word_embed_weight_word.npy'
w2v_content_word_model = 'data/my_w2v/train_word.model'

#data
train_data_cut_hdf ='data/cache/train_cut_word.hdf'
test_data_cut_hdf ='data/cache/test_cut_word.hdf'

