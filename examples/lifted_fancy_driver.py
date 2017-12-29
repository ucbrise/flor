#!/usr/bin/env python3
import project

from crawl import tr_crawl, te_crawl
from clean import clean
from train_model import train
from test_model import test

project.ground_client('git')
project.jarvisFile('lifted_fancy_driver.py')

training_tweets = project.Sample(0.8, 'training_tweets.csv', batch=True, times=3, to_csv=True)
testing_tweets = project.Sample(1.0, 'testing_tweets.csv', batch=True, times=1, to_csv=True)

cleaner = project.Action(clean, [training_tweets, testing_tweets])
clean_training_tweets = project.Artifact('clean_training_tweets.pkl', cleaner)
clean_testing_tweets = project.Artifact('clean_testing_tweets.pkl', cleaner)

trainer = project.Action(train, [clean_training_tweets])
intermediary = project.Artifact('intermediary.pkl', trainer)

tester = project.Action(test, [intermediary, clean_testing_tweets])
model_accuracy = project.Artifact('model_accuracy.txt', tester)

model_accuracy.pull()
model_accuracy.plot()