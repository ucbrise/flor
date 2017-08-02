#!/usr/bin/env python3
""" crawler.py
To run:
    crawler.py tr
    crawler.py te

The 'tr' flag crawls training tweets.
The 'te' flag crawls testing tweets.

Output:
    training_tweets.csv
    testing_tweets.csv

    ...Depending on the flag.

Output CSV schema:
    id : integer, primary key
    tweet: string, the text of the tweet
    place: string, usually CITY STATE but not consistently
    city: string, city
    country: string, Full name of country
    code: string, two-character country code
"""
import tweepy
import os.path
import sys
from credentials import keychain

abspath = os.path.dirname(os.path.abspath(__file__))

filename = ""
tweet_num = 0
tweet_list = []

# This array tells the crawler to find tweets from anywhere in the world.
anyloc = [-180, -90, 180, 90]

flag = sys.argv[1]
if flag == "tr":
    filename = "/training"
    tweet_num = 60000
elif flag == "te":
    filename = "/testing"
    tweet_num = 20000
else:
    # Unrecognized flag, exist with error.
    sys.exit(1)

"""
Returns api object
"""
def authenticate():
    consumer_key    = keychain["consumer_key"]
    consumer_secret = keychain["consumer_secret"]

    access_token    = keychain["access_token"]
    access_token_st = keychain["access_token_st"]

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_st)

    return tweepy.API(auth)

"""
Function to write the contents of tweet_list to a csv file
"""
def write2file():
    with open(abspath + filename + "_tweets.csv", 'w') as f:
        for text in tweet_list:
            f.write(text)

"""
Class used for streaming twitter
"""
class MyStreamListener(tweepy.StreamListener):
    
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
    
    def on_error(self, status_code):
        if status_code == 420:
            print("error")
            return False
    
    def on_status(self, status):
        self.num_tweets += 1

        idnum = str(self.num_tweets)
        tweettext = status.text.replace(',', '.').replace('\n', ' ')
        
        if status.place is None:
            self.num_tweets -= 1
            return True
        
        placefullname = str(status.place.full_name).replace(',', '').replace('\n', ' ')
        city = str(status.place.name).replace(',', '').replace('\n', ' ')
        country = str(status.place.country).replace(',', '').replace('\n', ' ')
        countrycode = str(status.place.country_code).replace(',', '').replace('\n', ' ')

        # Prepend a primary key, and replace commas in the tweet to periods
        # Convert new lines to spaces, because newline signals a new tweet
        text = ", ".join([idnum, tweettext, placefullname, city, country, countrycode]).replace('\n', '').replace('\r', '') + "\n"

        # Save to memory
        tweet_list.append(text)
        
        if(self.num_tweets >= tweet_num):
            # Before halting, write to file
            write2file()
            return False
        else:
            return True

def main():
    if os.path.isfile(abspath + filename + "_tweets.csv"):
        # A tweet csv file exists, so lets exist to avoid waste and expensive computation
        # However, it may be better to copy and paste the file to update the timestamp for make
        sys.exit(0)
    api = authenticate()
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    myStream.filter(locations=anyloc)

if __name__ == "__main__":
    main()
