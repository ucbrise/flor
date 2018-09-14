#!/usr/bin/env python3

import pandas as pd
from shared import params

import flor
import pickle


@flor.func
# this needs to be updated
def split(tweets_loc, frac, seed, tweet_df, test_df, **kwargs):
    tweet = pd.read_csv(tweets_loc, **params)
    tweet_count = len(tweet)
    tr_tweets_count = int(frac * tweet_count)

    # Shuffle the dataframe
    tweet = tweet.sample(frac=1, random_state=seed)

    # Split the df
    tr_tweet_df = tweet.iloc[:tr_tweets_count, :]
    te_tweet_df = tweet.iloc[tr_tweets_count:, :]

    tr_tweet_df.to_pickle(tweet_df)
    te_tweet_df.to_pickle(test_df)