#/usr/bin/env python
# coding=utf-8
import os
import sys
import numpy as np
import pandas as pd
import jieba
import re 

sys.path.append('utils/')
import config

#读取停用词
jieba.load_userdict(config.jieba_dict)
stopwords = [line.strip() for line in open(config.stopwords_path, 'r').readlines()]

#stopwords=[]


def clean_str(x):
    #数据清洗
    return x

def cut_single(x,cut_char_level):
    #分词
    x = clean_str(x)
    res = []
    if cut_char_level:
        setence_seged = list(x.strip())

    else:
        setence_seged = jieba.cut(x.strip())

    for word in setence_seged:
        if word not in stopwords:
            res.append(word)
            
    return res


def cut_word(path,cut_char_level):
    #读取数据，并分词  数据格式为df
    data = pd.read_csv(path,encoding='utf8', names=['id', 'review', 'label'])[1:]

    data['review'] = data['review'].astype(str)
    data['review_cut'] = data['review'].map(lambda x: cut_single(x,config.cut_char_level))
    print('cut done')
    return data
    
def read_cut(path,data_cut_hdf):
    #读取分词后的df
    if not os.path.exists(data_cut_hdf):
        data = cut_word(path,config.cut_char_level)
        data.to_hdf(data_cut_hdf, "data")
    data = pd.read_hdf(data_cut_hdf)
    return data
if __name__ == '__main__':
    path = config.origin_csv
    #read_data(path)
    cut_word(path)
