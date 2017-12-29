#!/usr/bin/env python3
import project

project.ground_client('git')

from crawl import tr_crawl
do_tr_crawl = project.Action(tr_crawl)
training_tweets = project.Artifact('training_tweets.csv', do_tr_crawl)

from clean import clean
do_tr_clean = project.Action(clean, [training_tweets])
clean_training_tweets = project.Artifact('clean_training_tweets.pkl', do_tr_clean)

from train_model import train
do_train = project.Action(train, [clean_training_tweets])
intermediary = project.Artifact('intermediary.pkl', do_train)

from crawl import te_crawl
do_te_crawl = project.Action(te_crawl)
testing_tweets = project.Artifact('testing_tweets.csv', do_te_crawl)

do_te_clean = project.Action(clean, [testing_tweets])
clean_testing_tweets = project.Artifact('clean_testing_tweets.pkl', do_te_clean)

from test_model import test
do_test = project.Action(test, [intermediary, clean_testing_tweets])
model_accuracy = project.Artifact('model_accuracy.txt', do_test)

model_accuracy.pull()
model_accuracy.plot()