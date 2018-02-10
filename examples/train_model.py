#!/usr/bin/env python3
# Hello world!
""" train_model.py
To run:
    train_model.py

Output:
    intermediary.pkl

intermediary.pkl is a python dictionary with the following keys, values:
{
    "vectorizer" : a scikit-learn vectorizer for text data,
    "country_dict" : a dictionary for converting between country code and integer,
    "classifier" : a scikit-learn classifier (multinomial-naive-bayes)
}

"""
import pandas as pd
import numpy as np
import os, pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from shared import params, relevant_attributes
import jarvis

abspath = os.path.dirname(os.path.abspath(__file__))

@jarvis.func
def train(tweet_df, alpha):
    intermediary = {}

    tweet_df = tweet_df.loc[:, relevant_attributes]

    # Convert string country code to integer country code
    country_codes = set([i for i in tweet_df["code"]])
    country_dict = {}
    for idx, code in enumerate(country_codes):
        country_dict[code] = idx

    intermediary["country_dict"] = country_dict

    def convert_to_int(country_string):
        return country_dict[country_string]

    tweet_df.loc[:, "code"] = tweet_df.loc[:, "code"].apply(convert_to_int)

    ## Convert tweet to bag of words for learning

    # Tokenize Text
    vectorizer = TfidfVectorizer()
    #vectorizer = CountVectorizer()

    X_train = vectorizer.fit_transform(tweet_df["tweet"])

    intermediary["vectorizer"] = vectorizer

    X_train_label = np.array(tweet_df["code"])

    # Train a classifier
    clf = MultinomialNB(alpha=alpha).fit(X_train, X_train_label)

    intermediary["classifier"] = clf

    return intermediary


def oldtrain(in_artifacts, out_artifacts):
    intermediary = {}

    in_artifact = in_artifacts[0]
    out_artifact = out_artifacts[0]

    with open(abspath + '/' + in_artifact.getLocation(), 'rb') as f:
        tweet_df = pickle.load(f)

    # Select a relevant subset of features
    tweet_df = tweet_df[relevant_attributes]

    # Convert string country code to integer country code
    country_codes = set([i for i in tweet_df["code"]])
    country_dict = {}
    for idx, code in enumerate(country_codes):
        country_dict[code] = idx

    intermediary["country_dict"] = country_dict
        
    def convert_to_int(country_string):
        return country_dict[country_string]

    tweet_df["code"] = tweet_df["code"].apply(convert_to_int)

    ## Convert tweet to bag of words for learning

    # Tokenize Text
    count_vect = CountVectorizer()
    X_train = count_vect.fit_transform(tweet_df["tweet"])

    intermediary["vectorizer"] = count_vect

    X_train_label = np.array(tweet_df["code"])

    # Train a classifier
    clf = MultinomialNB().fit(X_train, X_train_label)

    intermediary["classifier"] = clf

    with open(abspath + '/' + out_artifact.getLocation(), 'wb') as f:
        pickle.dump(intermediary, f)
