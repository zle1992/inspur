#!/usr/bin/python
# -*- coding: utf-8 -*-


batch_size = 128
number_classes = 2
w2v_vec_dim = 256
word_maxlen = 40

cut_char_level = False
model_dir = 'data/model_dir'
jieba_dict = 'data/jieba/jieba_dict.txt'
stopwords_path = 'data/jieba/stops.txt'
origin_csv = 'data/training-inspur.csv'
test_csv = 'submit/Preliminary-texting.csv'

#w2v
word_embed_weight = 'data/my_w2v/word_embed_weight_word.npy'
w2v_content_word_model = 'data/my_w2v/train_word.model'

#data
train_data_cut_hdf ='data/cache/train_cut_word.hdf'
test_data_cut_hdf ='data/cache/test_cut_word.hdf'
#自定义词典+停用词过滤
# data_hdf = 'data/atec_nlp_sim_train2.hdf'
# word_embed_weight = 'data/word_embed_weight_2.npy'
# w2v_content_word_model = 'data/train_w2v2.model'