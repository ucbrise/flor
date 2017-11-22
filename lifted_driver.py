#!/usr/bin/env python3
import jarvis
import random

jarvis.groundClient('git')
jarvis.jarvisFile('lifted_driver.py')


training_tweets = jarvis.Artifact('training_tweets.csv')

from clean import clean
do_tr_clean = jarvis.Action(clean, [training_tweets])
clean_training_tweets = jarvis.Artifact('clean_training_tweets.pkl', do_tr_clean)

@jarvis.func
def sample(tweet_df, frac, seed):
    return tweet_df.sample(frac=frac, random_state=seed)

do_tr_sample = jarvis.Action(sample, [clean_training_tweets, 0.8, [2, 5, 42]])
sampled_training_tweets = jarvis.Artifact('sampled_training_tweets.pkl', do_tr_sample)

from train_model import train
do_train = jarvis.Action(train, [sampled_training_tweets])
intermediary = jarvis.Artifact('intermediary.pkl', do_train)

testing_tweets = jarvis.Artifact('testing_tweets.csv')

do_te_clean = jarvis.Action(clean, [testing_tweets])
clean_testing_tweets = jarvis.Artifact('clean_testing_tweets.pkl', do_te_clean)

from test_model import test
do_test = jarvis.Action(test, [intermediary, clean_testing_tweets])
model_accuracy = jarvis.Artifact('model_accuracy.txt', do_test)

model_accuracy.pull()
model_accuracy.plot()