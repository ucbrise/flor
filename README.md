# Jarvis

Old Documentation.

## Environment
Python 3.5 or later.

## Python Packages
pip3 install ...
```
airflow
celery
mysqlclient
numpy
pandas
scikit-learn
tweepy
tweet-preprocessor
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
You will need to edit `credentials.py` to use your Twitter account. Follow the steps below to get your consumer key and access token.

1. Go to Twitter and sign in
2. Go to Twitter Application Management and create a new app
3. Fill in the form and create your Twitter application
4. Get your Consumer API key
5. You can create your Access Token and Access Token Secret by clicking on the button "Create my access token"

Enter this information into the `credentials.py` file, but keep that information private.

To run:
```
airflow backfill jarvisworkflow -s 2017-07-02
```
If the date entered is earlier than TODAY, the workflow will run immediately. Otherwise, it will run as scheduled. For more detailed information about how to schedule airflow workflows, please visit https://airflow.incubator.apache.org/tutorial.html

The classification accuracy score for this toy workflow will appear in *stdout.txt*

## Parallelize with Celery
This section assumes you read and ran the instructions in the previous sections successfully.

By default, Airflow will execute DAG tasks sequentially, and one at a time. To execute multiple tasks in parallel, satisfying the dependency constraints of the DAG, we will use Celery and RabbitMQ.

### Install and Run RabbitMQ
Visit the [RabbitMQ website](http://www.rabbitmq.com/download.html) for download and installation instructions.

After installing RabbitMQ, start the RabbitMQ server in detached mode:
```
sudo rabbitmq-server -detached
```
### Install and Configure MySQL
Airflow will need MySQL to use Celery. After installing MySQL, create a user (e.g. `airflow`) and a database (e.g. `airflow`). Grant all privileges for your new user on the new database. 

### Ubuntu Packages
After installing MySQL, install the following package:
```
sudo apt install libmysqlclient-dev
```

### Edit the Airflow Configuration File
Change directory to the AIRFLOW_HOME, and edit `airflow.cfg`. All the following changes apply to the `airflow.cfg` file.

1. Set the executor:
```
executor = CeleryExecutor
```
2. Set the SQL Alchemy Connection (replacing newuser, newpassword, and newdatabase with the names you chose during the "Install and Configure MySQL" step):
```
sql_alchemy_conn = mysql://newuser:newpassword@localhost/newdatabase
```
3. Do not pickle. Pickling in Airflow raises exceptions with Python 3.x:
```
donot_pickle = True
```
4. Change the broker URL for celery:
```
broker_url = amqp://guest:guest@localhost:5672
```
5. Change the result backend for celery (replacing newuser, newpassword, and newdatabase with the names you chose during the "Install and Configure MySQL" step):
```
celery_result_backend = db+mysql://newuser:newpassword@localhost:3306/newdatabase
```

### Let Airflow Initialize the MySQL Database
```
airflow initdb
```

### Run The Airflow Worker
To Parallelize Airflow, you will need at least two instances of the console or command line. 

In console 1, enter:
```
airflow worker
```
You may get an error message: `OSError: [Errno 98] Address already in use`, but this Error can be ignored for now.

In Console 2, enter:
```
airflow backfill jarvisworkflow -s 2017-07-02
```

Jarvis should now run in parallel.

