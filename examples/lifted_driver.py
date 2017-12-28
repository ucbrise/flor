#!/usr/bin/env python3
import jarvis
import numpy as np
import pandas as pd

jarvis.groundClient('git')
jarvis.jarvisFile('lifted_driver.py')


training_tweets = jarvis.Artifact('training_tweets.csv')

from clean import clean
do_tr_clean = jarvis.Action(clean, [training_tweets])
clean_training_tweets = jarvis.Artifact('clean_training_tweets.pkl', do_tr_clean)

alpha = jarvis.Literal(np.linspace(0.0, 1.0, 11).tolist(), 'alpha')
alpha.forEach()

from train_model import train
do_train = jarvis.Action(train, [clean_training_tweets, alpha])
intermediary = jarvis.Artifact('intermediary.pkl', do_train)

testing_tweets = jarvis.Artifact('testing_tweets.csv')

do_te_clean = jarvis.Action(clean, [testing_tweets])
clean_testing_tweets = jarvis.Artifact('clean_testing_tweets.pkl', do_te_clean)

from test_model import test
do_test = jarvis.Action(test, [intermediary, clean_testing_tweets])
model_accuracy = jarvis.Artifact('model_accuracy.txt', do_test)

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



