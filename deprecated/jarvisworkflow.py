#!/usr/bin/env python3
import airflow
from airflow import DAG
import os
from airflow.operators import BashOperator,PythonOperator
from datetime import datetime, timedelta

abspath = os.path.dirname(os.path.abspath(__file__))

default_args = {
    'owner': 'Rolando',
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(2),
    'email': ['rogarcia@berkeley.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('jarvisworkflow', default_args=default_args)

t1 = BashOperator(
    task_id='train',
    bash_command='python3 ' + abspath + '/crawler.py tr',
    dag=dag)

t2 = BashOperator(
    task_id='test',
    bash_command='python3 ' + abspath + '/crawler.py te',
    dag=dag)

t3 = BashOperator(
    task_id='cleantr',
    bash_command='python3 ' + abspath + '/cleaner.py tr',
    dag=dag)
t3.set_upstream(t1)

t4 = BashOperator(
    task_id='cleante',
    bash_command='python3 ' + abspath + '/cleaner.py te',
    dag=dag)
t4.set_upstream(t2)

t5 = BashOperator(
    task_id='predict',
    bash_command='python3 ' + abspath + '/predictor.py',
    dag=dag)
t5.set_upstream(t3)

t6 = BashOperator(
    task_id='validate',
    bash_command='python3 ' + abspath + '/validator.py',
    dag=dag)
t6.set_upstream([t4, t5])

t7 = BashOperator(
    task_id='verify',
    bash_command='python3 ' + abspath + '/verify.py',
    dag=dag)
t7.set_upstream(t6)

t8 = BashOperator(
    task_id='deploy',
    bash_command='python ' + abspath + '/deploy.py',
    dag=dag)
t8.set_upstream(t7)
