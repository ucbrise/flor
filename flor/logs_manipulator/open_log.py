import datetime
import json
import re
import uuid

import git

from flor.constants import *
from flor.face_library.flog import Flog
from flor.stateful import put, start
from flor.utils import cond_mkdir, refresh_tree, cond_rmdir


class OpenLog:

    def __init__(self, name, depth_limit=0):
        start()
        self.name = name
        cond_mkdir(os.path.join(FLOR_DIR, name))
        refresh_tree(FLOR_CUR)
        open(os.path.join(FLOR_CUR, name), 'a').close()

        log_file = open(Flog.__get_current__(), 'a')

        if depth_limit is not None:
            put('depth_limit', depth_limit)

        session_start = {'session_start': format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}

        # MAC address
        MAC_addr = {'MAC_address': ':'.join(re.findall('..?', '{:x}'.format(uuid.getnode())))}
        session_start.update(MAC_addr)


        # Get EC2 instance type
        try:
            import boto3
            ec2 = boto3.resource('ec2')
            for i in ec2.instances.all():
                session_start.update({'EC2_instance_type': i.instance_type})
        except:
            session_start.update({'EC2_instance_type': 'None'})

        # User info from Git
        class GitConfig(git.Repo):
            def __init__(self, *args, **kwargs):
                #
                # Work around the GitPython issue #775
                # https://github.com/gitpython-developers/GitPython/issues/775
                #
                self.git_dir = os.path.join(os.getcwd(), ".git")
                git.Repo.__init__(self, *args, **kwargs)

        r = GitConfig().config_reader()
        user_name = r.get_value('user', 'name')
        user_email = r.get_value('user', 'email')
        session_start.update({'git_user_name': user_name})
        session_start.update({'git_user_email': user_email})


        # System's userid
        import getpass
        user_id = getpass.getuser()
        session_start.update({'user_id': user_id})


        # Reliable timestamp from network server
        def get_ntp_time(host='time.nist.gov'):
            # Adapted from https://www.mattcrampton.com/blog/query_an_ntp_server_from_python/
            # and https://gist.github.com/guneysus/9f85ab77e1a11d0eebdb
            import socket
            from socket import AF_INET, SOCK_DGRAM
            import struct
            import time

            port = 123
            buf = 1024
            address = (host, port)
            msg = '\x1b' + 47 * '\0'

            # reference time (in seconds since 1900-01-01 00:00:00)
            time1970 = 2208988800  # 1970-01-01 00:00:00

            # connect to server
            client = socket.socket(AF_INET, SOCK_DGRAM)
            client.sendto(msg.encode('utf-8'), address)
            msg, address = client.recvfrom(buf)

            if msg:
                # timestamp: seconds since epoch
                timestamp = struct.unpack('!12I', msg)[10]
                timestamp -= time1970
                local_time = time.ctime(timestamp).replace('  ', ' ')
                utc_time = time.strftime('%a %b %d %X %Y %Z', time.gmtime(timestamp))
                return timestamp, local_time, utc_time

        timestamp, local_time, utc_time = get_ntp_time()

        session_start.update({'timestamp': timestamp})
        session_start.update({'local_time': local_time})
        session_start.update({'UTC_time': utc_time})

        log_file.write(json.dumps(session_start) + '\n')

        log_file.flush()
        log_file.close()

    def exit(self):
        log_file = open(Flog.__get_current__(), 'a')
        session_end = {'session_end': format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
        log_file.write(json.dumps(session_end) + '\n')
        log_file.flush()

        refresh_tree(FLOR_CUR)
        cond_rmdir(MODEL_DIR)

        log_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()