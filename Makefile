CONDA36_ENV ?= py36

deploy : validate version.v intermediary.pkl deploy.py
	python deploy.py

test : model_accuracy.txt

train : intermediary.pkl

validate : model_accuracy.txt shared.py intermediary.pkl clean_testing_tweets.pkl validate.py
	source activate $(CONDA36_ENV) && \
	python validate.py && \
	source deactivate && \
	test -f deployflag.txt

model_accuracy.txt : intermediary.pkl clean_testing_tweets.pkl shared.py test_model.py
	source activate $(CONDA36_ENV) && \
	python test_model.py && \
	source deactivate

intermediary.pkl : clean_training_tweets.pkl shared.py train_model.py
	source activate $(CONDA36_ENV) && \
	python train_model.py && \
	source deactivate

clean_testing_tweets.pkl : testing_tweets.csv shared.py cleaner.py
	source activate $(CONDA36_ENV) && \
	python cleaner.py te && \
	source deactivate

clean_training_tweets.pkl : training_tweets.csv shared.py cleaner.py
	source activate $(CONDA36_ENV) && \
	python cleaner.py tr && \
	source deactivate


testing_tweets.csv : credentials.py crawler.py
	source activate $(CONDA36_ENV) && \
	python crawler.py te  && \
	source deactivate

training_tweets.csv : credentials.py crawler.py
	source activate $(CONDA36_ENV) && \
	python crawler.py tr && \
	source deactivate

clean :
	rm -f *.pkl *.txt

.PHONY : clean train test validate deploy