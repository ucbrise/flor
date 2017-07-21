#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os, pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from shared import params, relevant_attributes

abspath = os.path.dirname(os.path.abspath(__file__))

with open(abspath + '/country_dict.pkl', 'rb') as f:
    country_dict = pickle.load(f)

with open(abspath + '/testing_tweets.pkl', 'rb') as f:
    test_df = pickle.load(f)

with open(abspath + '/stdout.txt', 'r') as f:
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

with open(abspath + "/deployflag.txt", "w") as f:
    if mostCommonClassFreq < accuracy:
        # model passes the test
        f.write("True")
    else:
        # model fails the test
        f.write("False")
    f.write("\n%.6f" % mostCommonClassFreq)
