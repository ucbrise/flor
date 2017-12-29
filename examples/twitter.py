#!/usr/bin/env python3
import jarvis

ex = jarvis.Experiment('twitter')

ex.groundClient('git')

training_tweets = ex.artifact('training_tweets.csv')

from clean import clean
do_tr_clean = ex.action(clean, [training_tweets])
clean_training_tweets = ex.artifact('clean_training_tweets.pkl', do_tr_clean)

from train_model import train
do_train = ex.action(train, [clean_training_tweets, 0.0001])
intermediary = ex.artifact('intermediary.pkl', do_train)

testing_tweets = ex.artifact('testing_tweets.csv')

do_te_clean = ex.action(clean, [testing_tweets])
clean_testing_tweets = ex.artifact('clean_testing_tweets.pkl', do_te_clean)

from test_model import test
do_test = ex.action(test, [intermediary, clean_testing_tweets])
model_accuracy = ex.artifact('model_accuracy.txt', do_test)

model_accuracy.pull()
model_accuracy.plot()