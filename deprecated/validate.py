#!/usr/bin/env python3
""" validate.py

To run:
    validate.py

Output:
    deployflag.txt
        line 1: either True or False
        line 2: floating point value in the range [0.0, 1.0] denoting the frequency of the most common class

"""
import pandas as pd
import numpy as np
import os, pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from shared import params, relevant_attributes

abspath = os.path.dirname(os.path.abspath(__file__))


# country_dict
with open(abspath + '/intermediary.pkl', 'rb') as f:
    intermediary = pickle.load(f)
    country_dict = intermediary["country_dict"]

with open(abspath + '/clean_testing_tweets.pkl', 'rb') as f:
    test_df = pickle.load(f)

with open(abspath + '/model_accuracy.txt', 'r') as f:
    content = f.readlines()
    accuracy = float(content[0].strip())

test_df = test_df[relevant_attributes]

def special_convert_to_int(country_string):
    if country_string in country_dict:
        return country_dict[country_string]
    else:
        return -1

test_df["code"] = test_df["code"].apply(special_convert_to_int)

# Ignore countries unseen in training (only 12 instances out of 20k)
test_df = test_df[test_df["code"] != -1]

X_test_label = test_df["code"]
countOfCommonClass = X_test_label[X_test_label == X_test_label.mode()[0]].size
mostCommonClassFreq = countOfCommonClass / X_test_label.size


wroteFile = False
with open(abspath + "/deployflag.txt", "w") as f:
    if mostCommonClassFreq < accuracy:
        wroteFile = True
        # model passes the test
        f.write("True")
    else:
        # model fails the test
        f.write("False")
    f.write("\n%.6f" % mostCommonClassFreq)

if not wroteFile:
    os.remove(abspath + "/deployflag.txt")