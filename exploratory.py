# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:02:31 2021

@author: Guillaume
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import re
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

train_path = 'D:/Guillaume/NLP_disaster/train.csv'
test_path ='D:/Guillaume/NLP_disaster/test.csv'
submission_path = 'D:/Guillaume/NLP_disaster/sample_submission.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_sample = pd.read_csv(submission_path)
#%%
train_df.head(5)
print("Total number of rows in train dataset are ",train_df.shape[0],'and total number of columns in train dataset are',train_df.shape[1])
print("Total number of rows in test dataset are ",test_df.shape[0],'and total number of columns in test dataset are',test_df.shape[1])
train_df.info()
train_df.isnull().sum()

# drop location and keyword column
train_df = train_df.drop(['location','keyword'],axis=1)
test_df = test_df.drop(['location','keyword'],axis=1)

real_tweets = len(train_df[train_df["target"] == 1])
real_tweets_percentage = real_tweets/train_df.shape[0]*100
fake_tweets_percentage = 100-real_tweets_percentage

print("Real tweets percentage: ",real_tweets_percentage)
print("Fake tweets percentage: ",fake_tweets_percentage)

length_train = train_df['text'].str.len() 
length_test = test_df['text'].str.len() 
plt.hist(length_train, label="train_tweets") 
plt.hist(length_test, label="test_tweets") 
plt.legend() 
plt.show()
non_disaster_tweets = train_df[train_df['target'] == 1]['text']
disaster_tweets = train_df[train_df['target'] ==1 ]['text']
for i in range(1,10):
    print(disaster_tweets[i])
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 5])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(disaster_tweets))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Disaster Tweets',fontsize=40);

wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(non_disaster_tweets))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Non Disaster Tweets',fontsize=40);

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Applying the cleaning function to both test and train datasets
train_df['text'] = train_df['text'].apply(lambda x: clean_text(x))
test_df['text'] = test_df['text'].apply(lambda x: clean_text(x))
# have list of word
tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')
train_df['text'] = train_df['text'].apply(lambda x:tokenizer.tokenize(x))
test_df['text'] = test_df['text'].apply(lambda x:tokenizer.tokenize(x))
train_df['text'].head()

def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words 
train_df['text'] = train_df['text'].apply(lambda x : remove_stopwords(x))
test_df['text'] = test_df['text'].apply(lambda x : remove_stopwords(x))
test_df.head()

# lemmatization eg: changed, changing  vhanges → change
lem = WordNetLemmatizer()
def lem_word(x):
    return [lem.lemmatize(w) for w in x]



train_df['text'] = train_df['text'].apply(lem_word)
test_df['text'] = test_df['text'].apply(lem_word)



def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

train_df['text'] = train_df['text'].apply(lambda x : combine_text(x))
test_df['text'] = test_df['text'].apply(lambda x : combine_text(x))


#CountVectorizer is used to transform a given text into a vector on the basis of the frequency(count) of each word that occurs in the entire text.It involves counting the number of occurences each words appears in a document(text)
count_vectorizer = CountVectorizer()
train_vector = count_vectorizer.fit_transform(train_df['text'])
test_vector = count_vectorizer.transform(test_df['text'])
print(train_vector[0].todense())
# Term Frequency-Inverse document frequency.It is a techinque to quantify a word in documents,we generally compute a weight to each word which signifies the importance of the word which signifies the importance of the word in the document and corpus
tfidf = TfidfVectorizer(min_df = 2,max_df = 0.5,ngram_range = (1,2))
train_tfidf = tfidf.fit_transform(train_df['text'])
test_tfidf = tfidf.transform(test_df['text'])

## xgb
xgb_param = xgb.XGBClassifier(max_depth=5,n_estimators=500,colsample_bytree=0.8,nthread=10,learning_rate=0.05)
scores_vector = model_selection.cross_val_score(xgb_param,train_vector,train_df['target'],cv=5,scoring='f1')


scores_tfidf = model_selection.cross_val_score(xgb_param,train_tfidf,train_df['target'],cv=5,scoring='f1')
scores_tfidf
## multinomial naive bayes

mnb = MultinomialNB(alpha = 2.0)
scores_vector = model_selection.cross_val_score(mnb,train_vector,train_df['target'],cv = 10,scoring = 'f1')
print("score:",scores_vector)
scores_tfidf = model_selection.cross_val_score(mnb,train_tfidf,train_df['target'],cv = 10,scoring = 'f1')
print("score of tfidf:",scores_tfidf)
## LogisticRegression
lg = LogisticRegression(C = 1.0)
scores_vector = model_selection.cross_val_score(lg, train_vector, train_df["target"], cv = 5, scoring = "f1")
print("score:",scores_vector)
scores_tfidf = model_selection.cross_val_score(lg, train_tfidf, train_df["target"], cv = 5, scoring = "f1")
print("score of tfidf:",scores_tfidf)

# pred
mnb.fit(train_tfidf, train_df["target"])
y_pred = mnb.predict(test_tfidf)

submission_df2 = pd.DataFrame({'Id':test_df['id'],'target':y_pred})


submission_df2.to_csv('submission_df2.csv',index=False)





