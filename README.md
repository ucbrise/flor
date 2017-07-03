# Jarvis

## Environment
Python 3.5 or later.

## Python Packages
pip3 install ...
```
airflow
numpy
pandas
scikit-learn
tweepy
```
For detailed Airflow installation instructions, please visit https://airflow.incubator.apache.org/installation.html

It is possible that the airflow executable will fail to appear in an executable PATH. In that case:
``` 
ln -sf ~/.local/lib/python3.5/site-packages/airflow/bin/airflow /usr/bin/airflow
```
But change the command to use absolute paths and the correct python version.

To test your airflow installation:
```
airflow initdb
```
## Jarvis Download and Installation
```
mkdir ~/airflow/dags
cd ~/airflow/dags
git clone git@github.com:ucbrise/jarvis.git
cd jarvis
python3 jarvisworkflow.py
```
To check that the installation was successful:
```
airflow list_dags
```
You should see `jarvisworkflow` in the list.

## Running Jarvis
You will need to edit `credentials.py` to use your Twitter account. Follow the steps below to get your consumer key and acces token.

1. Go to Twitter and sign in.
2. Go to Twitter Application Management and create a new app
3. Fill in the form and create your Twitter application
4. Get your Consumer API key
5. You can crete your Access Token and Access Token Secret by clicking on the button "Create my access token"

Enter this information into the `credentials.py` file, but keep that information private.

To run:
```
airflow backfill jarvisworkflow -s 2017-07-02
```
If the data entered is earlier than TODAY, the workflow will run immediately. Otherwise, it will run as scheduled. For more detailed information about how to schedule airflow workflows, please visit https://airflow.incubator.apache.org/tutorial.html

The classification accuracy score for this toy workflow will appear in *stdout.txt*
