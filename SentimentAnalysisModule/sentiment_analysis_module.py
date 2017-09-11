import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk import ngrams
import string
import random
from sklearn.utils import shuffle
from os import listdir
from os.path import isfile, join
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.stem import *
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle
from nltk.classify import ClassifierI
import os

def processSentence(sentence, tokenizer):
    words = tokenizer.tokenize(sentence)
    words = [word.lower() for word in words]
    for i in range(len(words)):
        if words[i] in keys:
            words[i] = dictionary[words[i]]
    new_sentence = " ".join(words)
    return new_sentence    



current_dir = os.path.dirname(__file__)
relative_path = "Classifiers"
abs_dir_path = os.path.join(current_dir, relative_path)



with open(os.path.join(abs_dir_path, "dictionary.pickle"), "rb") as f:
    dictionary = pickle.load(f)

with open(os.path.join(abs_dir_path, "vectorizer.pickle"), "rb") as f:
    vectorizer = pickle.load(f)

keys = dictionary.keys()  
tweetTokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)


with open(os.path.join(abs_dir_path, "clf1.pickle"), "rb") as f:
    clf1 = pickle.load(f)



with open(os.path.join(abs_dir_path, "clf2.pickle"), "rb") as f:
    clf2 = pickle.load(f) 


with open(os.path.join(abs_dir_path, "clf3.pickle"), "rb") as f:
    clf3 = pickle.load(f) 


with open(os.path.join(abs_dir_path, "clf4.pickle"), "rb") as f:
    clf4 = pickle.load(f) 


with open(os.path.join(abs_dir_path, "clf5.pickle"), "rb") as f:
    clf5 = pickle.load(f) 


with open(os.path.join(abs_dir_path, "clf6.pickle"), "rb") as f:
    clf6 = pickle.load(f) 



def classify(sentence):
    new_sentence = processSentence(sentence,tweetTokenizer)
    tf_idf = vectorizer.transform([new_sentence])
    return clf4.predict(tf_idf)[0]
