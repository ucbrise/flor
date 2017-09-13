#!/usr/bin/env python3
""" cleaner.py
"""
import pandas as pd
import html.parser
import preprocessor as twpre
import os, pickle, sys
from shared import params

abspath = os.path.dirname(os.path.abspath(__file__))

html_parser = html.parser.HTMLParser()

"""
Clean takes one in artifact and outputs one out artifact
"""
def clean(in_artifacts, out_artifacts, out_types):
    in_artifact = in_artifacts[0]
    out_artifacts = out_artifacts[0]
    tweet_df = pd.read_csv(abspath + '/' + in_artifact.getLocation('r'), **params)
    tweet_df["tweet"] = tweet_df["tweet"].apply(html_parser.unescape)
    tweet_df["tweet"] = tweet_df["tweet"].apply(twpre.tokenize)
    with open(abspath + '/'  + out_artifacts.getLocation('w'), 'wb') as f:
        pickle.dump(tweet_df, f)
