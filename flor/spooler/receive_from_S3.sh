#!/bin/bash

START_TIME=$(date +%s)

# Receive each file over the network from S3
aws s3 cp s3://flor/$2/data/$3 $1

END_TIME=$(date +%s)
TOT_TIME=$(expr $END_TIME - $START_TIME)
