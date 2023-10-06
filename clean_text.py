# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 19:12:45 2020

@author: jagma
"""
# import nltk
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# stop_words=stopwords('en')

def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    s=str(message) 
    s=re.sub("\\+", " ",s)
    s=re.sub("/+", " ",s)
    s=s.replace("\'+","'")
    s=s.replace("\"", "")
    s=s.replace("[","")
    s=s.replace(":", "")
    s=s.replace(":", "")
    s=s.replace("?", "")
    s=s.replace("(","")
    s=s.replace(")", "")
    s=s.replace("|", "")
    s=re.sub('\.+',' ',s)
    
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    # if gram > 1:
    #     w = []
    #     for i in range(len(words) - gram + 1):
    #         w += [' '.join(words[i:i + gram])]
    #         print(w)
    #     w=' '.join(w)
    #     return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]  
    words=' '.join(words)
    return words


