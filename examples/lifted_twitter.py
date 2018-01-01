#!/usr/bin/env python3
import jarvis
import numpy as np

ex = jarvis.Experiment("lifted_twitter")
ex.groundClient('git')

tweets = ex.artifact('tweets.csv')

frac = ex.literal(0.75, 'frac')
split_seed = ex.literal(42, 'split_seed')

from split import split
do_split = ex.action(split, [tweets, frac, split_seed])
training_tweets = ex.artifact('training_tweets.pkl', do_split)
testing_tweets = ex.artifact('testing_tweets.pkl', do_split)

alpha = ex.literal(np.linspace(0.0, 1.0, 11).tolist(), 'alpha')
alpha.forEach()

from train_model import train
do_train = ex.action(train, [training_tweets, alpha])
intermediary = ex.artifact('intermediary.pkl', do_train)

from test_model import test
do_test = ex.action(test, [intermediary, testing_tweets])
model_accuracy = ex.artifact('model_accuracy.txt', do_test)

columnArtifacts = {'model_accuracy': model_accuracy,
                   'model': intermediary}

df = model_accuracy.parallelPull(manifest=columnArtifacts)
best_intermediary = df.loc[df['model_accuracy'].idxmax()]['model']

country_dict = best_intermediary['country_dict']
classifier = best_intermediary['classifier']
vectorizer = best_intermediary['vectorizer']

code_dict = {}

for kee in country_dict:
    code_dict[country_dict[kee]] = kee

while True:
    tweet = input("What's on your mind? ")
    if tweet == 'nothing':
        break
    tweet_vec = vectorizer.transform(np.array([tweet,]))
    country_id = classifier.predict(tweet_vec)
    print("Predicted country of origin: {}\n".format(code_dict[country_id[0]]))
