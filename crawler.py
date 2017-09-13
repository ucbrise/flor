import pandas as pd
import numpy as np

def crawl(in_artifacts, out_artifacts, out_types, data_source):
	destination = out_artifacts[0].getLocation()
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

	df = pd.read_csv(data_source, **params)
	df.to_csv(destination, index=False, header=False)

def tr_crawl(in_artifacts, out_artifacts, out_types):
	crawl(in_artifacts, out_artifacts, out_types, 
		'deprecated/training_tweets.csv')

def te_crawl(in_artifacts, out_artifacts, out_types):
	crawl(in_artifacts, out_artifacts, out_types, 
		'deprecated/testing_tweets.csv')