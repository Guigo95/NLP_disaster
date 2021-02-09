# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:37:53 2021

@author: Guillaume
"""
from keras.optimizers import Adam

from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import random
import os

from data_prepare import read_glove_vecs,sentences_to_indices, clean_text
from data_prepare import convert_examples_to_features, create_tokenizer_from_hub_module, convert_text_to_examples
from utils import f1_m

import matplotlib.pyplot as plt


#%% prepare data

train_path = 'C:/Users/Guillaume/DL/project/NLP_disaster/data/train.csv'
test_path ='C:/Users/Guillaume/DL/project/NLP_disaster/data/test.csv'
submission_path = 'D:/Guillaume/NLP_disaster/sample_submission.csv'
save_dir = 'D:/Guillaume/NLP_disaster/'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_sample = pd.read_csv(submission_path)

# data preparation
train_df = train_df.drop(['location','keyword'],axis=1)
test_df = test_df.drop(['location','keyword'],axis=1)
train_df['text'] = train_df['text'].apply(lambda x: clean_text(x))
test_df['text'] = test_df['text'].apply(lambda x: clean_text(x))
text = train_df['text'].to_list()
text = [ phrase.split() for phrase in text]
l_all = [len(el) for el in text]
Tx = max(l_all)
X = train_df['text'].to_list()
Y =  train_df['target'].to_list()
ind = [i for i in range(len(X)-1)]
random.seed(1)
random.shuffle(ind)
train_index = ind[0:6500] 
val_index = ind[6500:-1]

es=EarlyStopping(monitor='val_f1_m', mode ='max', patience=15, verbose=0, baseline=None, restore_best_weights=True)
#%% basic lstm and lstm with attention / glove embedding

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('D:/Guillaume/NLP_disaster/glove.6B.50d.txt')
X_indices = sentences_to_indices(X, word_to_index,  max_len =Tx)


X_train_indices = np.asarray(X_indices[train_index])
Y_train_oh =   np.asarray(Y[train_index])
X_val_indices =  np.asarray(X_indices[val_index])
Y_val_oh =  np.asarray(Y[val_index])

#%% model lstm 
from model_basic_lstm import model_lstm

fold_name = 'basic_lstm_1'

m_lstm = model_lstm((Tx,), word_to_vec_map, word_to_index)
m_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])#%%

model_json = m_lstm.to_json()
with open(save_dir+fold_name +'.json', 'w') as json_file:
    json_file.write(model_json)
    
checkpoint = ModelCheckpoint(save_dir+fold_name +'.h5', 
							monitor='val_f1_m', verbose=1, 
							save_best_only=True, mode='max')
callbacks_list = [es,checkpoint]
history = m_lstm.fit(X_train_indices, Y_train_oh, validation_data= (X_val_indices, Y_val_oh), epochs = 50, batch_size = 32, shuffle=True, callbacks = callbacks_list)
# prediction
m_lstm.load_weights(save_dir + fold_name+".h5") 

# test on val
pred = m_lstm.predict(X_val_indices)
pred = pred.reshape(-1)
pred[pred>.5] = 1
pred[pred<=.5] = 0
pred = pred.astype('int64') 
F1_LSTM = f1_score(pred,Y_val_oh)
print('F1 score LSTM: ', F1_LSTM)


#%% model lstm with attention
from model_lstm_attention import model_attention
fold_name = 'attention_1'

m_attention = model_attention((Tx,), word_to_vec_map, word_to_index)
m_attention.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])#%%

model_json = m_attention.to_json()
with open(save_dir+fold_name +'.json', 'w') as json_file:
    json_file.write(model_json)
    
checkpoint = ModelCheckpoint(save_dir+fold_name +'.h5', 
							monitor='val_f1_m', verbose=1, 
							save_best_only=True, mode='max')
callbacks_list = [es,checkpoint]
history = m_attention.fit(X_train_indices, Y_train_oh, validation_data= (X_val_indices, Y_val_oh), epochs = 50, batch_size = 32, shuffle=True, callbacks = callbacks_list)
# prediction
m_attention.load_weights(save_dir + fold_name+".h5") 

# test on val
pred = m_attention.predict(X_val_indices)
pred = pred.reshape(-1)
pred[pred>.5] = 1
pred[pred<=.5] = 0
pred = pred.astype('int64') 
F1_attention = f1_score(pred,Y_val_oh)
print('F1 score LSTM with attention: ', F1_attention)

#%% bert model
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
tokenizer = create_tokenizer_from_hub_module(bert_path)

# Convert data to InputExample format
train_examples = convert_text_to_examples(X[train_index], Y[train_index])
val_examples = convert_text_to_examples(X[val_index], Y[val_index])
# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_labels 
) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=Tx)
(val_input_ids, val_input_masks, val_segment_ids, val_labels
) = convert_examples_to_features(tokenizer, val_examples, max_seq_length=Tx)
#%%
from model_bert import build_model
m_bert = build_model(Tx)
fold_name = 'bert_1'


m_bert.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])

model_json = m_bert.to_json()
with open(save_dir+fold_name +'.json', 'w') as json_file:
    json_file.write(model_json)
  
checkpoint = ModelCheckpoint(save_dir+fold_name +'.h5', 
							monitor='val_f1_m', verbose=1, 
							save_best_only=True, mode='max')
callbacks_list = [es,checkpoint]
history = m_bert.fit([train_input_ids, train_input_masks, train_segment_ids], train_labels, 
                     validation_data= ([val_input_ids, val_input_masks, val_segment_ids], val_labels), epochs = 50, batch_size = 32, shuffle=True, callbacks = callbacks_list)
# prediction
m_bert.load_weights(save_dir + fold_name+".h5") 

# test on val
post_save_preds = m_bert.predict([val_input_ids[0:100], 
                                val_input_masks[0:100], 
                                val_segment_ids[0:100]])

#F1_bert = f1_score(post_save_preds, val_labels)
#print('F1 score bert: ', F1_bert)

#%% Submission kaggle
X_test = np.asarray(test_df['text'].to_list())
#Y_test_oh =  np.asarray(test_df['target'].to_list())

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len =Tx)

pred = model.predict(X_test_indices)
pred = pred.reshape(-1)
pred[pred>.5] = 1
pred[pred<=.5] = 0

pred = pred.astype('int64') 

submission_df2 = pd.DataFrame({'Id':test_df['id'],'target':pred})


submission_df2.to_csv('submission_df2_lstm.csv',index=False)