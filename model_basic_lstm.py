# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:09:34 2021

@author: Guillaume
"""

from keras.layers import Bidirectional, Input, LSTM
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.models import  Model
from utils import pretrained_embedding_layer
import numpy as np



def model_lstm(input_shape, word_to_vec_map, word_to_index):

    sentence_indices = Input(shape = input_shape, dtype =np.int32)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)   
    X = Bidirectional(LSTM(128, return_sequences=True))(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=sentence_indices, outputs=X)
    

    
    return model


