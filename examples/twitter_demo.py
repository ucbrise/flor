#!/usr/bin/env python3
import jarvis
import numpy as np

ex = jarvis.Experiment("twitter_demo")
ex.groundClient('ground')

tweets = ex.artifact('tweets.csv')

frac = ex.literal(0.75, 'frac')
split_seed = ex.literal(42, 'split_seed')

from split import split
do_split = ex.action(split, [tweets, frac, split_seed])
training_tweets = ex.artifact('training_tweets.pkl', do_split)
testing_tweets = ex.artifact('testing_tweets.pkl', do_split)

alpha = ex.literal(np.linspace(0.0, 1.0, 8).tolist(), 'alpha')
alpha.forEach()

from train_model import train
do_train = ex.action(train, [training_tweets, alpha])
model = ex.artifact('model.pkl', do_train)

from test_model import test
do_test = ex.action(test, [model, testing_tweets])
model_accuracy = ex.artifact('model_accuracy.txt', do_test)

columnArtifacts = {'model_accuracy': model_accuracy,
                   'model': model}

#model_accuracy.parallelPull(manifest=columnArtifacts)
model_accuracy.pull(manifest=columnArtifacts)
