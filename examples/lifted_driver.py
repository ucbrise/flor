#!/usr/bin/env python3
import project
import numpy as np
import pandas as pd

project.groundClient('git')
project.jarvisFile('lifted_driver.py')


training_tweets = project.Artifact('training_tweets.csv')

from clean import clean
do_tr_clean = project.Action(clean, [training_tweets])
clean_training_tweets = project.Artifact('clean_training_tweets.pkl', do_tr_clean)

alpha = project.Literal(np.linspace(0.0, 1.0, 11).tolist(), 'alpha')
alpha.forEach()

from train_model import train
do_train = project.Action(train, [clean_training_tweets, alpha])
intermediary = project.Artifact('intermediary.pkl', do_train)

testing_tweets = project.Artifact('testing_tweets.csv')

do_te_clean = project.Action(clean, [testing_tweets])
clean_testing_tweets = project.Artifact('clean_testing_tweets.pkl', do_te_clean)

from test_model import test
do_test = project.Action(test, [intermediary, clean_testing_tweets])
model_accuracy = project.Artifact('model_accuracy.txt', do_test)

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



