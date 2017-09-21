#!/usr/bin/env python3
import jarvis

from crawler import tr_crawl, te_crawl
from cleaner import clean
from train_model import train
from test_model import test

tr_crawler = jarvis.Action(func=tr_crawl)
training_tweets = jarvis.Artifact(loc='training_tweets.csv',
	typ="data", parent=tr_crawler)

te_crawler = jarvis.Action(func=te_crawl)
testing_tweets = jarvis.Artifact(loc='testing_tweets.csv',
	typ="data", parent=te_crawler)

tr_cleaner = jarvis.Action(func=clean, 
	in_artifacts=[training_tweets])
clean_training_tweets = jarvis.Artifact(loc='clean_training_tweets.pkl',
	typ='data', parent=tr_cleaner)

te_cleaner = jarvis.Action(func=clean,
	in_artifacts=[testing_tweets])
clean_testing_tweets = jarvis.Artifact(loc='clean_testing_tweets.pkl',
	typ='data', parent=te_cleaner)

trainer = jarvis.Action(func=train,
	in_artifacts=[clean_training_tweets])
intermediary = jarvis.Artifact(loc='intermediary.pkl',
	typ='model', parent=trainer)

tester = jarvis.Action(func=test,
	in_artifacts=[intermediary, clean_testing_tweets])
model_accuracy = jarvis.Artifact(loc='model_accuracy.txt',
	typ='metadata', parent=tester)

model_accuracy.pull()

