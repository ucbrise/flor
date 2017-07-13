#!/usr/bin/env python3
import pandas as pd
import html.parser
import preprocessor as twpre
import os, pickle, sys
from shared import params

abspath = os.path.dirname(os.path.abspath(__file__))

flag = sys.argv[1]
if flag == "tr":
    filename = "/training"
elif flag == "te":
    filename = "/testing"
else:
    sys.exit(1)

tweet_df = pd.read_csv(abspath + filename + '_tweets.csv', **params)

html_parser = html.parser.HTMLParser()

tweet_df["tweet"].apply(html_parser.unescape)
tweet_df["tweet"].apply(twpre.tokenize)

with open(abspath + filename + '_tweets.pkl', 'wb') as f:
    pickle.dump(tweet_df, f)