#/usr/bin/env python
# coding=utf-8
import os
import sys
import numpy as np
import pandas as pd
sys.path.append('utils/')
import config
from w2v import load_my_train_w2v,load_pre_train_w2v

#laod vocab
vocab, embed_weights = load_my_train_w2v(config.origin_csv)



def padding_id(ids, padding_token=0, padding_length=None):
    if len(ids) > padding_length:
        return ids[:padding_length]
    else:
        return ids + [padding_token] * (padding_length - len(ids))

def word2id(contents, word_voc):
    ''' contents  list
    '''
#     contents = str(contents)
#     contents = contents.split()

    ids = [word_voc[c] if c in word_voc else len(word_voc) for c in contents]

    return padding_id(ids, padding_token=0, padding_length=config.word_maxlen)

def data2id(data):
    data['review_id'] = data['review_cut'].map(lambda x: word2id(x, vocab))
    return data

