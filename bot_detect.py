# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:37:00 2020

@author: jag
"""
import pandas as pd
from clean_text import process_message
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib

# df=pd.read_excel('bot_comment_dataset/final/spam_comments.xlsx')
# print(df.info())
# df.dropna(subset=[],inplace=True)
# #print(len(df))

# comments=[]
# spam_flag=[]
# for x in range(len(df['CONTENT'])):
#     text=process_message(df['CONTENT'][x])
#     spam_flag.append(df['CLASS'][x])
#     comments.append(text)
    
# comment_np=np.array(comments)
# x_train,x_test,y_train,y_test=train_test_split(comments,spam_flag,test_size=0.2,random_state=42)

# vectorizer=CountVectorizer()
# x_train=vectorizer.fit_transform(x_train)
# joblib.dump(vectorizer,'vectorizer.vect')
# naive_classifier=MultinomialNB()
# naive_classifier.fit(x_train,y_train)
# joblib.dump(naive_classifier,'lang_detect.model')

vectorizer=joblib.load('vectorizer.vect')
NB=joblib.load('lang_detect.model')

test=vectorizer.transform([process_message('i post funny memes. check my porfile and drop a follow')])
print(NB.predict(test))


# x_test=vectorizer.transform(x_test)

# prediction=NB.predict(x_test)
# from sklearn.metrics import confusion_matrix
# print('confusion_matrix')
# print(confusion_matrix(y_test,prediction))
# print(f1_score(y_test,prediction,average='weighted'))
# np.ndarray.dump(naive_classifier,open('lang_detect.model','wb'))









