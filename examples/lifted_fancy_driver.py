#!/usr/bin/env python3
import jarvis

from crawl import tr_crawl, te_crawl
from clean import clean
from train_model import train
from test_model import test

jarvis.ground_client('git')
jarvis.jarvisFile('lifted_fancy_driver.py')

training_tweets = jarvis.Sample(0.8, 'training_tweets.csv', batch=True, times=3, to_csv=True)
testing_tweets = jarvis.Sample(1.0, 'testing_tweets.csv', batch=True, times=1, to_csv=True)

cleaner = jarvis.Action(clean, [training_tweets, testing_tweets])
clean_training_tweets = jarvis.Artifact('clean_training_tweets.pkl', cleaner)
clean_testing_tweets = jarvis.Artifact('clean_testing_tweets.pkl', cleaner)

trainer = jarvis.Action(train, [clean_training_tweets])
intermediary = jarvis.Artifact('intermediary.pkl', trainer)

tester = jarvis.Action(test, [intermediary, clean_testing_tweets])
model_accuracy = jarvis.Artifact('model_accuracy.txt', tester)

model_accuracy.pull()
model_accuracy.plot()