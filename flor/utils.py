import shutil
from flor.constants import *
import os


def cond_mkdir(path):
    """
    Mkdir if not exists
    :param path:
    :return:
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def refresh_tree(path):
    """
    When finished, brand new directory root at path
        Whether or not it used to exist and was empty
    :param path:
    :return:
    """
    cond_rmdir(path)
    os.mkdir(path)


def cond_rmdir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


def check_flor_install():
    if not os.path.exists(os.path.join(FLOR_DIR, '.conda_map')):
        print("Flor hasn't been installed.")
        print("From Python: You may run the function flor.install()")
        print("From CLI: You may run the pyflor_install script")
        import sys
        sys.exit(0)

def write_debug_msg(msg):
    assert isinstance(msg, str)
    with open(os.path.join(FLOR_DIR, 'debug_msg.txt'), 'a') as f:
        f.write(msg + '\n')

def write_failure_msg(msg):
    assert isinstance(msg, str)
    with open(os.path.join(FLOR_DIR, 'failures.txt'), 'a') as f:
        f.write(msg + '\n')

# Reliable timestamp from network server
def get_timestamp(src='time.google.com'):
    # Adapted from https://www.mattcrampton.com/blog/query_an_ntp_server_from_python/
    # and https://gist.github.com/guneysus/9f85ab77e1a11d0eebdb
    import socket
    from socket import AF_INET, SOCK_DGRAM
    import struct
    import time

    port = 123
    buf = 1024
    address = (src, port)
    msg = '\x1b' + 47 * '\0'

    # reference time (in seconds since 1900-01-01 00:00:00)
    time1970 = 2208988800  # 1970-01-01 00:00:00
    timestamp = time.time()
    # connect to server
    try:
        socket.setdefaulttimeout(3)  # set timeout to 3s
        client = socket.socket(AF_INET, SOCK_DGRAM)
        client.sendto(msg.encode('utf-8'), address)
        msg, address = client.recvfrom(buf)

        if msg:
            # timestamp: seconds since epoch
            timestamp = struct.unpack('!12I', msg)[10] - time1970
    except socket.timeout:
        src = 'local'

    local_time = time.ctime(timestamp).replace('  ', ' ')
    utc_time = time.strftime('%a %b %d %X %Y %Z', time.gmtime(timestamp))

    return timestamp, local_time, utc_time, src