intermediary.pkl : shared.py clean_training_tweets.pkl train_model.py
	source activate py36 && python train_model.py && source deactivate

training_tweets.csv : credentials.py crawler.py
	source activate py36 && python crawler.py tr && source deactivate

deploy : validate version.v intermediary.pkl deploy.py
	python deploy.py

clean_testing_tweets.pkl : shared.py testing_tweets.csv cleaner.py
	source activate py36 && python cleaner.py te && source deactivate

clean_training_tweets.pkl : shared.py training_tweets.csv cleaner.py
	source activate py36 && python cleaner.py tr && source deactivate

model_accuracy.txt : shared.py intermediary.pkl clean_testing_tweets.pkl test_model.py
	source activate py36 && python test_model.py && source deactivate

train : intermediary.pkl

clean : 
	rm -f *.pkl *.txt

test : model_accuracy.txt

validate : shared.py model_accuracy.txt intermediary.pkl clean_testing_tweets.pkl validate.py
	source activate py36 && python validate.py && source deactivate && test -f deployflag.txt

testing_tweets.csv : credentials.py crawler.py
	source activate py36 && python crawler.py te && source deactivate

.PHONY : clean train test validate deploy
