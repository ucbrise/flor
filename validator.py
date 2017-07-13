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

## Convert tweet to bag of words for learning

# Tokenize Text
with open(abspath + '/vectorizer.pkl', 'rb') as f:
    count_vect = pickle.load(f)

clf = joblib.load(abspath + '/classifier.pkl')

## Now we test.
with open(abspath + '/testing_tweets.pkl', 'rb') as f:
    test_df = pickle.load(f)
    
test_df = test_df[relevant_attributes]

def special_convert_to_int(country_string):
    if country_string in country_dict:
        return country_dict[country_string]
    else:
        return -1

test_df["code"] = test_df["code"].apply(special_convert_to_int)

# Ignore countries unseen in training (only 12 instances out of 20k)
test_df = test_df[test_df["code"] != -1]

# Tokenize Text
X_test = count_vect.transform(test_df["tweet"])
X_test_label = np.array(test_df["code"].data)

score = clf.score(X_test, X_test_label)

with open(abspath + "/stdout.txt", "w") as f:
    f.write("Score: %.6f" % score)