#!/usr/bin/env python3
""" cleaner.py

To run:
    cleaner.py tr
    cleaner.py te

The 'tr' flag cleans training tweets.
The 'te' flag cleans testing tweets.

Output:
    clean_training_tweets.pkl
    clean_testing_tweets.pkl

    ...Depending on the flag.
    The output is a pandas dataframe with a schema of the Input CSV schema

Input CSV schema:
    id : integer, primary key
    tweet: string, the text of the tweet
    place: string, usually CITY STATE but not consistently
    city: string, city
    country: string, Full name of country
    code: string, two-character country code

"""
import pandas as pd
import html.parser
import preprocessor as twpre
import os, pickle, sys
from shared import params

abspath = os.path.dirname(os.path.abspath(__file__))

html_parser = html.parser.HTMLParser()

def clean(inputdf_column):
    inputdf_column.apply(html_parser.unescape)
    inputdf_column.apply(twpre.tokenize)

def main():
    flag = sys.argv[1]
    if flag == "tr":
        filename = "training"
    elif flag == "te":
        filename = "testing"
    else:
        sys.exit(1)

    tweet_df = pd.read_csv(abspath + '/' + filename + '_tweets.csv', **params)
    clean(tweet_df["tweet"])

    with open(abspath + '/clean_'  + filename + '_tweets.pkl', 'wb') as f:
        pickle.dump(tweet_df, f)

if __name__ == "__main__":
    main()
