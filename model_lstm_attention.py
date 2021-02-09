# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:09:34 2021

@author: Guillaume
"""
from keras.layers import Bidirectional,Input, LSTM,  Activation,  Layer
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.models import Model
import keras.backend as K
import numpy as np

from utils import pretrained_embedding_layer

def dot_product(x, kernel):
   
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()


def model_attention(input_shape, word_to_vec_map, word_to_index):

    sentence_indices = Input(shape = input_shape, dtype =np.int32)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)   
    X = Bidirectional(LSTM(128, return_sequences=True))(embeddings)
    X = Dropout(0.5)(X)
    X = attention()(X)
    X = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=sentence_indices, outputs=X)
       
    return model


