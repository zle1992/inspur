#/usr/bin/env python
# coding=utf-8
import os
import sys
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import pandas as pd
import jieba
sys.path.append('utils/')
import config

jieba.load_userdict(config.jieba_dict)


#stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
stopwords = ['?', ',', '。']


def cut_single(x):
    res = []
    setence_seged = jieba.cut(x.strip())
    for word in setence_seged:
        if word not in stopwords:
            res.append(word)
    return res


def make_w2v(path):
    if not os.path.exists(config.w2v_content_word_model):

        data1 = read_cut(config.origin_csv)
        data2 = read_cut(config.test_csv)
        content = list(data1.review_cut) + list(data2.review_cut)
        print('content len: ',len(content))
        model = Word2Vec(content, size=config.w2v_vec_dim, window=5, min_count=5,
                         )
        model.save(config.w2v_content_word_model)

    model = Word2Vec.load(config.w2v_content_word_model)

    weights = model.wv.syn0
    vocab = dict([(k, v.index + 1) for k, v in model.wv.vocab.items()])
    vocab['<-UNKNOW->'] = len(vocab) + 1
    embed_weights = np.zeros(shape=(weights.shape[0] + 2, weights.shape[1]))
    embed_weights[1:weights.shape[0] + 1] = weights
    unk_vec = np.random.random(size=weights.shape[1]) * 0.5
    pading_vec = np.random.random(size=weights.shape[1]) * 0
    embed_weights[weights.shape[0] + 1] = unk_vec - unk_vec.mean()
    embed_weights[0] = pading_vec

    np.save(config.word_embed_weight, embed_weights)
    print(embed_weights.shape)
    print('save embed_weights!')
    return vocab, embed_weights


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


def read_cut(path):

    data = pd.read_csv(path, encoding='ISO-8859-1',#encoding='utf8',
                       names=['id', 'review', 'label'])[1:]

    data['review'] = data['review'].astype(str)
    data['review_cut'] = data['review'].map(lambda x: cut_single(x))
    print('cut done')
    return data


def read_hdf(path):
    if not os.path.exists(config.data_hdf):
        data = read_data(path)
        data.to_hdf(config.data_hdf, "data")
    else:
        data = pd.read_hdf(config.data_hdf)
    return data


def read_data(path):
    data = read_cut(path)
    vocab, embed_weights = make_w2v(config.origin_csv)
    data['review_id'] = data['review_cut'].map(lambda x: word2id(x, vocab))
    return data


if __name__ == '__main__':
    path = config.origin_csv
    read_data(path)
