import os
import stat

from . import connect

sh_send_to_s3 = """
#!/bin/bash

START_TIME=$(date +%s)

# Send index file to be stored on S3
aws s3 cp "{home}/.flor/"$1/$2  s3://flor/$1/

# Send each data file to be stored on S3
for FILE in $(ls "{home}/.flor/$1/data")
do
   aws s3 cp "{home}/.flor/$1/data/"$FILE s3://flor/$1/data/
   rm "{home}/.flor/$1/data/"$FILE
done

END_TIME=$(date +%s)
TOT_TIME=$(expr $END_TIME - $START_TIME)
""".format(home=os.path.expanduser('~'))

sh_receive_index_from_s3 = """
#!/bin/bash

START_TIME=$(date +%s)

# Receive index file from S3
aws s3 cp s3://flor/$1/$2 {home}/.flor/$1/ 

END_TIME=$(date +%s)
TOT_TIME=$(expr $END_TIME - $START_TIME)
""".format(home=os.path.expanduser('~'))

sh_receive_from_s3 = """
#!/bin/bash

START_TIME=$(date +%s)

# Receive each data file from S3
aws s3 cp s3://flor/$2/data/$3 $1

END_TIME=$(date +%s)
TOT_TIME=$(expr $END_TIME - $START_TIME)
"""

flor_dir = os.path.expanduser(os.path.join('~', '.flor'))

send_path = os.path.join(flor_dir, 'send_to_S3.sh')
receive_index_path = os.path.join(flor_dir, 'receive_index_from_S3.sh')
receive_path = os.path.join(flor_dir, 'receive_from_S3.sh')

with open(send_path, 'w') as f:
    f.write(sh_send_to_s3)
os.chmod(send_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
with open(receive_index_path, 'w') as f:
    f.write(sh_receive_index_from_s3)
os.chmod(receive_index_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
with open(receive_path, 'w') as f:
    f.write(sh_receive_from_s3)
os.chmod(receive_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

__all__ = ['connect']
