import math
import os
import shutil


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


def fprint(dir_tree_list, device_id):
    root_path = os.path.sep + os.path.join(*dir_tree_list)
    def write(s):
        with open(os.path.join(root_path, "flor_output_{}.txt".format(device_id)), 'a') as f:
            f.write(s + '\n')

    return write


def get_partitions(iterator, num_gpu):
    """
    Returns at most num_gpu partitions.
    Balances desire to spread work evenly with the interest in using fewer GPUs if possible
    """
    work_per_gpu = math.ceil(len(iterator) / num_gpu)
    i = 0
    partitions = []
    while i * work_per_gpu < len(iterator):
        partitions.append(iterator[i * work_per_gpu: (i + 1) * work_per_gpu])
        i += 1
    return partitions
