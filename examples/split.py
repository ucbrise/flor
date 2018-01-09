#!/usr/bin/env python3

import pandas as pd
from shared import params

import jarvis


@jarvis.func
def split(tweets_loc, frac, seed):
    tweet_df = pd.read_csv(tweets_loc, **params)
    tweet_count = len(tweet_df)
    tr_tweets_count = int(frac * tweet_count)

    # Shuffle the dataframe
    tweet_df = tweet_df.sample(frac=1, random_state=seed)

    # Split the df
    tr_tweet_df = tweet_df.iloc[:tr_tweets_count, :]
    te_tweet_df = tweet_df.iloc[tr_tweets_count:, :]

    return tr_tweet_df, te_tweet_df
