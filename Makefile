training_tweets.csv:
	python3 crawler.py tr

testing_tweets.csv:
	python3 crawler.py te 

training_tweets.pkl: training_tweets.csv
	python3 cleaner.py tr

testing_tweets.pkl: testing_tweets.csv
	python3 cleaner.py te

intermediary.pkl: training_tweets.pkl
	python3 predictor.py

stdout.txt: intermediary.pkl testing_tweets.pkl
	python3 validator.py

deployfalg.txt: stdout.txt
	python3 verify.py

train: intermediary.pkl

validate: stdout.txt

deploy: deployfalg.txt
	python deploy.py

clean:
	rm *.pkl *.txt
