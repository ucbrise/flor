#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

abspath = os.path.dirname(os.path.abspath(__file__))

# Define the names of each column in the tweets file
attribute_names = []
attribute_names.append('id')
attribute_names.append('tweet')
attribute_names.append('place')
attribute_names.append('city')
attribute_names.append('country')
attribute_names.append('code')

# Define the data type of every element in a column
attribute_types = {
    'id': np.int32,
    'tweet': str,
    'place': str,
    'city': str,
    'country': str,
    'code': str
}

# Read the twitter data into a pandas dataframe
params = dict(header=None, names=attribute_names, dtype=attribute_types)
tweet_df = pd.read_csv(abspath + '/training_tweets.csv', **params)

# Select a relevant subset of features
relevant_attributes = ["tweet", "code"]
tweet_df = tweet_df[relevant_attributes]

# Convert string country code to integer country code
country_codes = set([i for i in tweet_df["code"]])
country_dict = {}
for idx, code in enumerate(country_codes):
    country_dict[code] = idx
    
def convert_to_int(country_string):
    return country_dict[country_string]

tweet_df["code"] = tweet_df["code"].apply(convert_to_int)

## Convert tweet to bag of words for learning

# Tokenize Text
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(tweet_df["tweet"])

X_train_label = np.array(tweet_df["code"].data)

# Train a classifier
clf = MultinomialNB().fit(X_train, X_train_label)

## Now we test.
test_df = pd.read_csv(abspath + '/testing_tweets.csv', **params)
test_df = test_df[relevant_attributes]

def special_convert_to_int(country_string):
    if country_string in country_dict:
        return convert_to_int(country_string)
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