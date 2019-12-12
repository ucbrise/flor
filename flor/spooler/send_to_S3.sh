#!/bin/bash
  
START_TIME=$(date +%s)

# Send each file to be stored on S3
for FILE in $(ls "~/.flor/$1/data")
do
   aws s3 cp "~/.flor/$1/data/"$FILE s3://flor/$1/data/
   rm "~/.flor/$1/data/"$FILE
done

END_TIME=$(date +%s)
TOT_TIME=$(expr $END_TIME - $START_TIME)
