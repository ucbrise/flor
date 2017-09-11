#!/usr/bin/env python3

import jarvis
import os, dill

abspath = os.path.dirname(os.path.abspath(__file__))

def crawl():
	import pandas as pd
	import numpy as np
	# Define the names of each column in the tweets file
	attribute_names = []
	attribute_names.append('id')
	attribute_names.append('tweet')
	attribute_names.append('place')
	attribute_names.append('city')
	attribute_names.append('country')
	attribute_names.append('code')

	# Define the data type of every element in a column
	attribute_types = {
	    'id': np.int32,
	    'tweet': str,
	    'place': str,
	    'city': str,
	    'country': str,
	    'code': str
	}

	# Read the twitter data into a pandas dataframe
	params = dict(header=None, names=attribute_names, dtype=attribute_types)

	return pd.read_csv('training_tweets.csv', **params)

crawler = jarvis.Task(func = crawl, preconditions=[], postconditions=[])

with open(abspath + '/workflow.pkl', 'wb') as f:
	dill.dump(crawler, f)

