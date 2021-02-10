# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:37:53 2021

@author: Guillaume Godefroy
Mddified from Simple Text Multi Classification Task Using Keras BERT, Analytics Vidhya and Sequence Models, Coursera
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score
from bert import bert_tokenization
import numpy as np
import pandas as pd
import random

from data_prepare import read_glove_vecs, sentences_to_indices, clean_text, bert_encode
from model_bert import bert_model
from model_basic_lstm import model_lstm
from model_lstm_attention import model_attention, attention
from utils import f1_m

# %% %%%%%  prepare data %%%%%

train_path = 'data/train.csv'
test_path = 'data/test.csv'
submission_path = 'results/sample_submission.csv'
save_dir = 'models/'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_sample = pd.read_csv(submission_path)

train_df = train_df.drop(['location', 'keyword'], axis=1)
test_df = test_df.drop(['location', 'keyword'], axis=1)
train_df['text'] = train_df['text'].apply(lambda x: clean_text(x))
test_df['text'] = test_df['text'].apply(lambda x: clean_text(x))
text = train_df['text'].to_list()
text = [phrase.split() for phrase in text]
l_all = [len(el) for el in text]
Tx = max(l_all)
X = train_df['text'].to_list()
Y = train_df['target'].to_list()
ind = [i for i in range(len(X) - 1)]
random.seed(1)
random.shuffle(ind)
train_index = ind[0:2000]
val_index = ind[6500:-1]

es = EarlyStopping(monitor='val_f1_m', mode='max', patience=15, verbose=0, baseline=None, restore_best_weights=True)

# %% %%%%% data for lstm and lstm + attention %%%%%

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('D:/Guillaume/NLP_disaster/glove.6B.50d.txt')
X_indices = sentences_to_indices(X, word_to_index, max_len=Tx)
X_train_indices = np.asarray(X_indices[train_index])
X_val_indices = np.asarray(X_indices[val_index])
Y_train_oh = np.asarray(Y)
Y_train_oh = Y_train_oh[train_index]
Y_val_oh = np.asarray(Y)
Y_val_oh = Y_val_oh[val_index]
X_test = np.asarray(test_df['text'].to_list())
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=Tx)
# %%  %%%%%   lstm %%%%%

fold_name = 'basic_lstm_1'

# model
m_lstm = model_lstm((Tx,), word_to_vec_map, word_to_index)
m_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])

# save
model_json = m_lstm.to_json()
with open(save_dir + fold_name + '.json', 'w') as json_file:
    json_file.write(model_json)

# training
checkpoint = ModelCheckpoint(save_dir + fold_name + '.h5', monitor='val_f1_m', verbose=1, save_best_only=True,
                             mode='max')
callbacks_list = [es, checkpoint]
#%%
history = m_lstm.fit(X_train_indices, Y_train_oh, validation_data=(X_val_indices, Y_val_oh), epochs=50, batch_size=32,
                     shuffle=True, callbacks=callbacks_list)

# prediction on validation set
m_lstm.load_weights(save_dir + fold_name + ".h5")
pred = m_lstm.predict(X_val_indices)
pred = pred.reshape(-1)
pred[pred > .5] = 1
pred[pred <= .5] = 0
pred = pred.astype('int64')
F1_LSTM = f1_score(pred, Y_val_oh)
print('F1 score LSTM: ', F1_LSTM)

# %% %%%%%  model lstm with attention %%%%%

fold_name = 'attention_1'

# model
m_attention = model_attention((Tx,), word_to_vec_map, word_to_index)
m_attention.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])

# save
model_json = m_attention.to_json()
with open(save_dir + fold_name + '.json', 'w') as json_file:
    json_file.write(model_json)

checkpoint = ModelCheckpoint(save_dir + fold_name + '.h5', monitor='val_f1_m', verbose=1, save_best_only=True,
                             mode='max')
callbacks_list = [es, checkpoint]
#%%
history = m_attention.fit(X_train_indices, Y_train_oh, validation_data=(X_val_indices, Y_val_oh), epochs=50,
                          batch_size=32, shuffle=True, callbacks=callbacks_list)

# prediction on validation set
m_attention.load_weights(save_dir + fold_name + ".h5")
pred = m_attention.predict(X_val_indices)
pred = pred.reshape(-1)
pred[pred > .5] = 1
pred[pred <= .5] = 0
pred = pred.astype('int64')
F1_attention = f1_score(pred, Y_val_oh)
print('F1 score LSTM with attention: ', F1_attention)

# %% Submission kaggle

fold_name = 'basic_lstm_1'
model = m_lstm  # m_attention
model.load_weights(save_dir + fold_name + ".h5")

pred = model.predict(X_test_indices)
pred = pred.reshape(-1)
pred[pred > .5] = 1
pred[pred <= .5] = 0

pred = pred.astype('int64')

submission_df2 = pd.DataFrame({'Id': test_df['id'], 'target': pred})

submission_df2.to_csv('submission_df2_lstm.csv', index=False)

# %%  %%%%%  Bert model %%%%%
fold_name = 'bert_model_1'

# prepare data
module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = bert_tokenization.FullTokenizer(
    vocab_file=vocab_file,
    do_lower_case=do_lower_case
)

X_train = bert_encode(train_df.text.values, tokenizer, max_len=Tx)
X_test = bert_encode(test_df.text.values, tokenizer, max_len=Tx)
Y_train = tf.keras.utils.to_categorical(train_df.target.values, num_classes=2)

# model
m_bert = bert_model(bert_layer, max_len=Tx)
m_bert.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[f1_m])
# save and load
model_json = m_bert.to_json()
with open(save_dir + fold_name + '.json', 'w') as json_file:
    json_file.write(model_json)

checkpoint = ModelCheckpoint(save_dir + fold_name + '.h5', monitor='val_f1_m', verbose=1, save_best_only=True,
                             mode='max')
callbacks_list = [es, checkpoint]
#%%
train_history = m_bert.fit(X_train, Y_train, validation_split=0.2, epochs=50, callbacks=callbacks_list,
                          batch_size=8, verbose=1)

# %% Submission kaggle
m_bert.load_weights(save_dir + fold_name + ".h5")

pred = m_bert.predict(X_test)

pred = np.array([np.argmax(pred_i) for pred_i in pred])

submission_df2 = pd.DataFrame({'Id': test_df['id'], 'target': pred[:]})

submission_df2.to_csv('submission_bert.csv', index=False)
