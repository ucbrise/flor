#!/usr/bin/env python
""" deploy.py

To run:
    deploy.py

Publishes model to Clipper, if the model passes validation test.

"""
import pandas as pd
import numpy as np
import os, pickle, sys
from clipper_admin import Clipper
import HTMLParser

abspath = os.path.dirname(os.path.abspath(__file__))

with open(abspath + '/intermediary.pkl', 'rb') as f:
    intermediary = pickle.load(f)
    global_country_dict = intermediary["country_dict"]
    global_count_vect = intermediary["vectorizer"]
    global_clf = intermediary["classifier"]

html_parser = HTMLParser.HTMLParser()

def clean(inputdf_column):
    inputdf_column.apply(html_parser.unescape)
    # inputdf_column.apply(twpre.tokenize)

def txt_to_country(tweet):
    tweetdf_col = pd.DataFrame(data=[tweet])[0]
    clean(tweetdf_col)
    clf = global_clf
    country_dict = global_country_dict
    count_vect = global_count_vect
    vect_df = count_vect.transform(tweetdf_col)
    country_int = clf.predict(vect_df)
    for countrycode in country_dict.keys():
        if country_dict[countrycode] == country_int:
            return countrycode
    return "ERR"

clipper_conn = Clipper("localhost")
clipper_conn.start()

if "jarvis" not in clipper_conn.get_all_apps():
    clipper_conn.register_application("jarvis", "txt_to_country", "strings", "ERROR", 1000000000)

with open(abspath + '/version.v', 'r') as f:
    content = f.readlines()
    versh = int(content[0].strip())

clipper_conn.deploy_predict_function("txt_to_country", versh, txt_to_country, "strings")

with open(abspath + '/version.v', 'w') as f:
    f.write(str(versh + 1))

