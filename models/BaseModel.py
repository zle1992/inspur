# -*- coding: utf-8 -*-
import os
import sys
from abc import ABCMeta, abstractmethod
from datetime import datetime

import pandas as pd
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Layer, Activation
from keras import initializers, regularizers, constraints
from keras import backend as K
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard, ReduceLROnPlateau
from sklearn.metrics import accuracy_score,confusion_matrix
from keras.layers import *


class TextModel(object):
    """abstract base model for all text classification model."""
   
    __metaclass__ = ABCMeta
    def __init__(self,  
        model_name,
        nb_epoch, 
        max_len, 
        embed_size, 
        batch_size,
        lr,
        kfold,
        word_embed_weight, 
        stack_path,
        model_dir,
        use_pretrained, 
        trainable,
        **kwargs
        ):
        """
        :param model_name: 模型名称
        :param nb_epoch: 迭代次数
        :param max_len:  规整化每个句子的长度
        :param embed_size: 词向量维度
        :param last_act: 最后一层的激活函数
        :param batch_size:
        :param optimizer: 优化器
        :param use_pretrained: 是否嵌入层使用预训练的模型
        :param trainable: 是否嵌入层可训练, 该参数只有在use_pretrained为真时有用
        :param kwargs
        """
        #self.kwargs = kwargs

        self.model_name = model_name
        self.nb_epoch =nb_epoch
        self.max_len =max_len
        self.embed_size = embed_size
        self.batch_size =batch_size
        self.lr = lr
        self.kfold =kfold
        self.word_embed_weight = word_embed_weight
        self.stack_path =stack_path
        self.model_dir = model_dir
        self.use_pretrained = use_pretrained
        self.trainable = trainable



        self.optimizer = "adam"   
        self.metrics = ['accuracy']
        self.loss = 'categorical_crossentropy'
       
        
        
        self.time = datetime.now().strftime('%y%m%d%H%M%S')

        #self.kwargs = kwargs
        # self.callback_list = []
     
        #self.is_kfold = kwargs.get('is_kfold', False)
         #kwargs.get('kfold', 5)
        # if self.is_kfold:
        #     self.bst_model_path_list = []
        # self.is_retrain = kwargs.get('is_retrain') if not self.trainable else False  # 当trainble 为False时才is_retrain 可用
        # self.use_new_vector = kwargs.get('use_new_vector')

    @abstractmethod
    def get_model(self,) -> Model:
        """定义一个keras net, compile it and return the model"""
        raise NotImplementedError
    def make_model(self):
        model_inputs ,model_outputs = self.get_model()
        model = Model(inputs=model_inputs, outputs=model_outputs)
        model.compile(loss=self.loss,optimizer=self.optimizer, metrics=self.metrics)
        #model.summary()
        return model
    @abstractmethod
    def _get_bst_model_path(self) -> str:
        """return a name which is used for save trained weights"""
        raise NotImplementedError
    

    def create_embedding(self, **kwargs):
        "Create embedding layer from a pretrained weights array or random"
        pretrained_weights = np.load(self.word_embed_weight)
        in_dim, out_dim = pretrained_weights.shape

        if self.use_pretrained :
            
            embedding = Embedding(in_dim, out_dim, weights=[
                              pretrained_weights], trainable=self.trainable, **kwargs)
        else:

            embedding = Embedding(in_dim, out_dim, **kwargs)
        return embedding




    def get_bst_model_path(self):
        dirname = self._get_model_path()
        path = os.path.join(dirname, self._get_bst_model_path())
        return path

    def _get_model_path(self):
        _module = self.__class__.__dict__.get('__module__')
        model_dir = "/".join(_module.split(".")[:-1])
        return model_dir
        
    def score(self,y_t, y_p):
        acc = accuracy_score(y_t, y_p)
        print(confusion_matrix(y_t, y_p))
        print('acc:',acc)
        return acc

    def _train(self,data,bst_model_path):
        x_train, y_train ,x_dev, y_dev = data
        model  = self.make_model()
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        change_lr=ReduceLROnPlateau(monitor='val_acc',mode = 'max',factor=0.1,epsilon=0.001,min_lr=0.0001,patience=1)
        K.set_value(model.optimizer.lr, self.lr)

        model.fit(x_train, y_train,
            epochs=self.nb_epoch,
            validation_data=(x_dev, y_dev),
            batch_size=self.batch_size,
            callbacks=[model_checkpoint,early_stopping,change_lr,
            # TensorBoard(log_dir='data/log_dir'),
             ],
        )
        
        model.load_weights(bst_model_path)
        model.compile(loss=self.loss,optimizer=self.optimizer, metrics=self.metrics)
        return model
    

    
    def single_train(self,data):
        x_train, y_train ,x_dev, y_dev = data
        bst_model_path = self.get_bst_model_path()
        model = self._train(data,bst_model_path)
        pred = model.predict(x_dev)
        y_t=np.argmax(y_dev, axis=1)
        y_p = np.argmax(pred, axis=1)
        acc = self.score(y_t,y_p)

    def _predict(self,data,bst_model_path):
        K.clear_session()
        X,_=data
        model  = self.make_model()
        model.load_weights(bst_model_path)
        model.compile(loss=self.loss,optimizer=self.optimizer, metrics=self.metrics)
        model.predict(X, batch_size=self.batch_size)
        test_pred = model.predict(X, batch_size=self.batch_size)
        return test_pred

    def single_predict(self,data):
        bst_model_path = self.get_bst_model_path()
        return self._predict(data,bst_model_path)

    def make_train_cv_data(self,data_all, kfolds=5):
        X_train, Y_train = data_all
        S_train = np.zeros((Y_train.shape[0], 4))
        train_df= pd.DataFrame()
        X, Y = X_train, Y_train
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=kfolds, shuffle=True,random_state=2018)
        k = 0
        a= []    
        for train_index, test_index in kf.split(Y):
            k += 1
            bst_model_path = self.get_bst_model_path()+"_%d.h5" % (k)
            

            x_train = X_train[train_index]
            x_dev = X_train[test_index]
            y_train=Y_train[train_index]
            y_dev = Y_train[test_index]
            print('kf: ', k)

            model_trained = self._train([x_train, y_train, x_dev, y_dev],bst_model_path)

            pred = model_trained.predict(x_dev)
            y_t=np.argmax(y_dev, axis=1)
            y_p = np.argmax(pred, axis=1)
            
            S_train[test_index, :3] = pred
            S_train[test_index,3] = y_t
            acc = self.score(y_t,y_p)
            a.append(acc)

        train_df['{0}_0'.format(self.model_name)] = S_train[:,0]
        train_df['{0}_1'.format(self.model_name)] = S_train[:,1]
        train_df['{0}_2'.format(self.model_name)] = S_train[:,2]
        train_df['label'] = S_train[:,3]


        print('acc  list',a)
        print('mean :',np.mean(np.array(a)))
        train_df.to_csv(self.stack_path+'train_%s.csv' % (k),
                        index=False, )
    def make_test_cv_data(self,data, kfolds=5):

        X, _ = data
        test_df = pd.DataFrame()
        pred_probs = np.zeros((X.shape[0],3))
        
        for kf in range(1, kfolds + 1):
            print('kf: ', kf)
            bst_model_path = self.get_bst_model_path()+"_%d.h5" % (kf)
        
            pred = self._predict(data,bst_model_path)
           
            pred_probs +=pred
        pred_probs /=kfolds
	
        print(pred_probs.shape)
        test_df['{0}_0'.format(self.model_name)] = pred_probs[:,0]
        test_df['{0}_1'.format(self.model_name)] = pred_probs[:,1]
        test_df['{0}_2'.format(self.model_name)] = pred_probs[:,2]
        test_df['label'] =  np.argmax(pred_probs, axis=1) +1
        test_df.to_csv(self.stack_path+'test_%s.csv' % (self.model_name),
                       index=False,)
        return pred_probs
