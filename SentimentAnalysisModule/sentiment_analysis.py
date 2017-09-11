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
from os import listdir
from os.path import isfile, join
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import cross_val_score


def getMisspellingDictionary(path):
    dictionary = {}
    keys = []
    value = ''
    with open(path) as f:
        for line in f:
            line = line.strip('\n\r ').lower()
            if line.startswith('$'):
                if keys:
                    for key in keys:
                        if key not in dictionary:
                            dictionary[key] = value
                keys = []
                value = line[1:]
            else:
                keys.append(line)
    return dictionary

def getAbbreviationsDictionary(path):
    dictionary = {}
    with open(path) as f:
        lines = [line.strip('\n\r ') for line in f]
        lines = [line.lower() for line in lines if line]
        for line in lines:
            split = line.split('\t')
            if split[0] not in dictionary:
                dictionary[split[0]] = split[1]
            
    return dictionary

def processSentence(sentence, tokenizer):
    words = tokenizer.tokenize(sentence)
    words = [word.lower() for word in words]
    for i in range(len(words)):
        if words[i] in keys:
            words[i] = dictionary[words[i]]
    new_sentence = " ".join(words)
    return new_sentence

def replaceSentences(data, tokenizer):
    new_data = []
    for sentence in data:
        new_sentence = processSentence(sentence, tokenizer)
        new_data.append(new_sentence)
    return new_data





df = pd.read_csv('movie-pang02.csv')
df = shuffle(df)
df = shuffle(df)
df = shuffle(df)
df = shuffle(df)

df = df[['text','class']]

train = df[:1000]
test = df[1000:]

X_train = train['text']
y_train = train['class']
X_test = test['text']
y_test = test['class']


abbreviationDictionary = getAbbreviationsDictionary('dictionary.txt')
misspellingDictionary = getMisspellingDictionary('birkbeck.txt')

dictionary = {**abbreviationDictionary, **misspellingDictionary}

##with open("./Classifiers/dictionary.pickle", "wb") as f:
##    pickle.dump(dictionary, f)

keys = dictionary.keys()  
tweetTokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

X_train = replaceSentences(X_train, tweetTokenizer)
X_test = replaceSentences(X_test, tweetTokenizer)

vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8,
                             sublinear_tf=True, use_idf=True,
                             stop_words='english')

train_tf_idf = vectorizer.fit_transform(X_train)

##with open("./Classifiers/vectorizer.pickle", "wb") as f:
##    pickle.dump(vectorizer, f)

test_tf_idf = vectorizer.transform(X_test)



clf1 = SVC()
clf1.fit(train_tf_idf,y_train)

##with open("./Classifiers/clf1.pickle", "wb") as f:
##    pickle.dump(clf1, f)

accuracy = clf1.score(test_tf_idf,y_test)
print("SVC: ",accuracy)



clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(train_tf_idf,y_train)

##with open("./Classifiers/clf2.pickle", "wb") as f:
##    pickle.dump(clf2, f)

accuracy = clf2.score(test_tf_idf,y_test)
print("KNN: ",accuracy)


clf3 = MultinomialNB()
clf3.fit(train_tf_idf,y_train)

##with open("./Classifiers/clf3.pickle", "wb") as f:
##    pickle.dump(clf3, f)

accuracy = clf3.score(test_tf_idf,y_test)
print("Multinomial Naive Bayes: ",accuracy)


clf4 = LinearSVC()
clf4.fit(train_tf_idf,y_train)

##with open("./Classifiers/clf4.pickle", "wb") as f:
##    pickle.dump(clf4, f)

accuracy = clf4.score(test_tf_idf,y_test)
print("LinearSVC: ", accuracy)


clf5 = SVC(kernel='linear')
clf5.fit(train_tf_idf,y_train)

##with open("./Classifiers/clf5.pickle", "wb") as f:
##    pickle.dump(clf5, f)

accuracy = clf5.score(test_tf_idf,y_test)
print("SVC(kernel=linear): ",accuracy)


clf6 = LogisticRegression()
clf6.fit(train_tf_idf,y_train)

##with open("./Classifiers/clf6.pickle", "wb") as f:
##    pickle.dump(clf6, f)

accuracy = clf6.score(test_tf_idf,y_test)
print("Logistic regression: ",accuracy)






