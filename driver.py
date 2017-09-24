#!/usr/bin/env python3
import jarvis

from crawler import tr_crawl, te_crawl
from cleaner import clean
from train_model import train
from test_model import test

tr_crawler = jarvis.Action(tr_crawl)
training_tweets = jarvis.Artifact('training_tweets.csv', tr_crawler)

te_crawler = jarvis.Action(te_crawl)
testing_tweets = jarvis.Artifact('testing_tweets.csv', te_crawler)

tr_cleaner = jarvis.Action(clean, [training_tweets])
clean_training_tweets = jarvis.Artifact('clean_training_tweets.pkl', tr_cleaner)

te_cleaner = jarvis.Action(clean, [testing_tweets])
clean_testing_tweets = jarvis.Artifact('clean_testing_tweets.pkl', te_cleaner)

trainer = jarvis.Action(train, [clean_training_tweets])
intermediary = jarvis.Artifact('intermediary.pkl', trainer)

tester = jarvis.Action(test, [intermediary, clean_testing_tweets])
model_accuracy = jarvis.Artifact('model_accuracy.txt', tester)

model_accuracy.pull()

