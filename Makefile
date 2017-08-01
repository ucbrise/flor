training_tweets.csv:
	source activate py36 && python3 crawler.py tr && source deactivate

testing_tweets.csv:
	source activate py36 && python3 crawler.py te  && source deactivate

training_tweets.pkl: training_tweets.csv
	source activate py36 && python3 cleaner.py tr && source deactivate

testing_tweets.pkl: testing_tweets.csv
	source activate py36 && python3 cleaner.py te && source deactivate

intermediary.pkl: training_tweets.pkl
	source activate py36 && python3 predictor.py && source deactivate

stdout.txt: intermediary.pkl testing_tweets.pkl
	source activate py36 && python3 validator.py && source deactivate

deployfalg.txt: stdout.txt
	source activate py36 && python3 verify.py && source deactivate

train: intermediary.pkl

validate: stdout.txt

deploy: deployfalg.txt
	python deploy.py

clean:
	rm -f *.pkl *.txt
