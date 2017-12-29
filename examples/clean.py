#!/usr/bin/env python3
""" cleaner.py
"""
import pandas as pd
import html.parser
import preprocessor as twpre
import os, pickle, sys
from shared import params
import jarvis

abspath = os.path.dirname(os.path.abspath(__file__))

html_parser = html.parser.HTMLParser()

@jarvis.func
def clean(tweets_df_loc):
    tweet_df = pd.read_csv(tweets_df_loc, **params)
    return tweet_df

def oldclean(in_artifacts, out_artifacts):
    if len(in_artifacts) == 1:
        in_artifact = in_artifacts[0]
        out_artifacts = out_artifacts[0]
        tweet_df = pd.read_csv(abspath + '/' + in_artifact.getLocation(), **params)
        tweet_df["tweet"] = tweet_df["tweet"].apply(html_parser.unescape)
        with open(abspath + '/'  + out_artifacts.getLocation(), 'wb') as f:
            pickle.dump(tweet_df, f)
    else:
        clean[1]([in_artifacts[0],], [out_artifacts[0],])
        clean[1]([in_artifacts[1],], [out_artifacts[1],])
