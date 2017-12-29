#!/usr/bin/env python3
import project

from crawl import tr_crawl, te_crawl
from clean import clean
from train_model import train
from test_model import test

tr_crawler = project.Action(tr_crawl)
training_tweets = project.Artifact('training_tweets.csv', tr_crawler)

te_crawler = project.Action(te_crawl)
testing_tweets = project.Artifact('testing_tweets.csv', te_crawler)

cleaner = project.Action(clean, [training_tweets, testing_tweets])
clean_training_tweets = project.Artifact('clean_training_tweets.pkl', cleaner)
clean_testing_tweets = project.Artifact('clean_testing_tweets.pkl', cleaner)

trainer = project.Action(train, [clean_training_tweets])
intermediary = project.Artifact('intermediary.pkl', trainer)

tester = project.Action(test, [intermediary, clean_testing_tweets])
model_accuracy = project.Artifact('model_accuracy.txt', tester)

model_accuracy.pull()

model_accuracy.plot()