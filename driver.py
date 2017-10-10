#!/usr/bin/env python3
import jarvis

from crawl import tr_crawl
do_tr_crawl = jarvis.Action(tr_crawl)
training_tweets = jarvis.Artifact('training_tweets.csv', do_tr_crawl)

from clean import clean
do_tr_clean = jarvis.Action(clean, [training_tweets])
clean_training_tweets = jarvis.Artifact('clean_training_tweets.pkl', do_tr_clean)

from train_model import train
do_train = jarvis.Action(train, [clean_training_tweets])
intermediary = jarvis.Artifact('intermediary.pkl', do_train)

from crawl import te_crawl
do_te_crawl = jarvis.Action(te_crawl)
testing_tweets = jarvis.Artifact('testing_tweets.csv', do_te_crawl)

do_te_clean = jarvis.Action(clean, [testing_tweets])
clean_testing_tweets = jarvis.Artifact('clean_testing_tweets.pkl', do_te_clean)

from test_model import test
do_test = jarvis.Action(test, [intermediary, clean_testing_tweets])
model_accuracy = jarvis.Artifact('model_accuracy.txt', do_test)

model_accuracy.pull()

model_accuracy.plot()