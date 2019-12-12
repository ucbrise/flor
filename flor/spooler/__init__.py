import os
import stat

from . import connect

sh_receive_from_s3 = """
#!/bin/bash

START_TIME=$(date +%s)

# Receive each file over the network from S3
aws s3 cp s3://flor/$2/data/$3 $1

END_TIME=$(date +%s)
TOT_TIME=$(expr $END_TIME - $START_TIME)
"""

sh_send_to_s3 = """
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
"""

flor_dir = os.path.expanduser(os.path.join('~', '.flor'))

send_path = os.path.join(flor_dir, 'send_to_S3.sh')
receive_path = os.path.join(flor_dir, 'receive_from_S3.sh')

with open(send_path, 'w') as f:
    f.write(sh_send_to_s3)
os.chmod(send_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
with open(receive_path, 'w') as f:
    f.write(sh_receive_from_s3)
os.chmod(receive_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

__all__ = ['connect']
