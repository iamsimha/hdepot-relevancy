#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from nltk.stem.porter import *
np.random.seed(123)

stemmer = PorterStemmer()
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}


def remove_stop_words(sent):
    stop_words = ['I', 'i', 'a','about','an','are','as','at','be','by','com','for','from','how','in','is','it','of','on','or','that','the','this','to','was','what','when','where','who','will','with','the','www']
    stop_words = dict(map(lambda x : (x,1), stop_words))
    result = []
    for word in sent.split(" "):
        if word not in stop_words:
            result.append(word)
    return " ".join(result)

def jaccard_similarity(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    common_words = sum(str2.find(word) > -1 for word in str1.split())
    total_words = len(set((str1+ str2).split()))
    return common_words/total_words

def common_words(str1, str2):
    str1 = str1.lower()
    str2 =str2.lower()
    return sum(str1.find(word) > -1 for word in str2.split())

def train_test_split(df):
    num_samples  = df.shape[0]
    train_ind = np.random.rand(num_samples) < 0.7
    return (df[train_ind], df[~train_ind])

def rmse(y, y_pred):
    return mean_squared_error(y, y_pred)**0.5


def split_dataset(df):
    train_data, rem_data = train_test_split(df)
    test_data, hold_out_data = train_test_split(rem_data)
    train, test, hold_out = dict(), dict(), dict()

    train['X'] = train_data.drop(['relevance'], axis=1)
    train['Y'] = train_data['relevance'].values
    test['X'] = test_data.drop(['relevance'], axis=1)
    test['Y'] = test_data['relevance'].values
    hold_out['X'] = hold_out_data.drop(['relevance'], axis=1)
    hold_out['Y'] = hold_out_data['relevance'].values
    return (train, test, hold_out)

def debug_model(df, y, ypred):
    total = 0
    for i in xrange(len(y)):
        if (y[i] - ypred[i]) ** 2 >=1.5:
            print y[i], ypred[i], df[i:i+1][['search_term', 'product_description', 'product_title']].values
            total += 1
            print
            print
        if total > 10   :
            break

# def str_stem(s):
#     if isinstance(s, str):
#         s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
#         #s = re.sub(r'( [a-z]+)([A-Z][a-z])', r'\1 \2', s)
#         s = s.lower()
#         s = s.replace("  "," ")
#         s = re.sub(r"([0-9]),([0-9])", r"\1\2", s)
#         s = s.replace(","," ")
#         s = s.replace("$"," ")
#         s = s.replace("?"," ")
#         s = s.replace("-"," ")
#         s = s.replace("//","/")
#         s = s.replace("..",".")
#         s = s.replace(" / "," ")
#         s = s.replace(" \\ "," ")
#         s = s.replace("."," . ")
#         s = re.sub(r"(^\.|/)", r"", s)
#         s = re.sub(r"(\.|/)$", r"", s)
#         s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
#         s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
#         s = s.replace(" x "," xbi ")
#         s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
#         s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
#         s = s.replace("*"," xbi ")
#         s = s.replace(" by "," xbi ")
#         s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
#         s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
#         s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
#         s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
#         s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
#         s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
#         s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
#         s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
#         s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
#         s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
#         s = s.replace("°"," degrees ")
#         s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
#         s = s.replace(" v "," volts ")
#         s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
#         s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
#         s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
#         s = s.replace("  "," ")
#         s = s.replace(" . "," ")
#         #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
#         s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
#         s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

#         s = s.lower()
#         s = s.replace("toliet","toilet")
#         s = s.replace("airconditioner","air condition")
#         s = s.replace("vinal","vinyl")
#         s = s.replace("vynal","vinyl")
#         s = s.replace("skill","skil")
#         s = s.replace("snowbl","snow bl")
#         s = s.replace("plexigla","plexi gla")
#         s = s.replace("rustoleum","rust oleum")
#         s = s.replace("whirpool","whirlpool")
#         s = s.replace("whirlpoolga", "whirlpool ga")
#         s = s.replace("whirlpoolstainless","whirlpool stainless")
#         return s
#     else:
#         return "null"

def str_stem(s):
    s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
    #s = re.sub(r'( [a-z]+)([A-Z][a-z])', r'\1 \2', s)
    s = (s.lower()
         .replace("  "," "))
    s = re.sub(r"([0-9]),([0-9])", r"\1\2", s)
    s = (s.replace(","," ")
        .replace("$"," ")
        .replace("?"," ")
        .replace("-"," ")
        .replace("//","/")
        .replace("..",".")
        .replace(" / "," ")
        .replace(" \\ "," ")
        .replace("."," . "))
    s = re.sub(r"(^\.|/)", r"", s)
    s = re.sub(r"(\.|/)$", r"", s)
    s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
    s = s.replace(" x "," xbi ")
    s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
    s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
    s = s.replace("*"," xbi ")
    s = s.replace(" by "," xbi ")
    s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
    s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
    s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
    #s = s.replace("°"," degrees ")
    s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    s = s.replace(" v "," volts ")
    s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
    s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
    s = s.replace("  "," ")
    s = s.replace(" . "," ")
    #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
    strlist = s.split(" ")
    s = (" ").join([str(strNum[z]) if z in strNum else z for z in strlist])
    s = (" ").join([stemmer.stem(z) for z in strlist])

    s = (s.replace("toliet","toilet")
         .replace("airconditioner","air condition")
         .replace("vinal","vinyl")
         .replace("vynal","vinyl")
         .replace("skill","skil")
         .replace("snowbl","snow bl")
         .replace("plexigla","plexi gla")
         .replace("rustoleum","rust oleum")
         .replace("whirpool","whirlpool")
         .replace("whirlpoolga", "whirlpool ga")
         .replace("whirlpoolstainless","whirlpool stainless"))
    return s