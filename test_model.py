#!/usr/bin/env python3
""" test_model.py

To run:
	test_model.py

Output:
	model_accuracy.txt
		line 1: floating point value in the range [0.0, 1.0]

"""
import pandas as pd
import numpy as np
import os, pickle, sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from shared import params, relevant_attributes

abspath = os.path.dirname(os.path.abspath(__file__))

def test(in_artifacts, out_artifacts, out_types):
    out_artifact = out_artifacts[0]

    with open(abspath + '/' + in_artifacts[0].getLocation(), 'rb') as f:
        intermediary = pickle.load(f)

    country_dict = intermediary["country_dict"]
    count_vect = intermediary["vectorizer"]
    clf = intermediary["classifier"]

    ## Now we test.
    with open(abspath + '/' + in_artifacts[1].getLocation(), 'rb') as f:
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
    X_test_label = np.array(test_df["code"])

    score = clf.score(X_test, X_test_label)

    with open(abspath + "/" + out_artifact.getLocation(), "w") as f:
        f.write("%.5f" % score)
    return os.path.basename(__file__)
