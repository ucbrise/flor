import math
import os
import shutil
import flor.common.copy


class PATH:
    def __init__(self, root_path, path_from_home):
        root_path = '~' if root_path is None else root_path
        self.path_from_home = path_from_home
        self.squiggles = os.path.join(root_path, path_from_home)
        if root_path == '~':
            self.absolute = os.path.join(os.path.expanduser('~'), path_from_home)
        else:
            self.absolute = os.path.join(os.path.abspath(root_path), path_from_home)


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
            f.write(str(s) + '\n')

    return write


def get_partitions(listlen, num_gpus):
    # Roundrobin allocation with pipelining
    partitions = [[] for _ in range(num_gpus)]
    for epoch in range(listlen):
        partitions[epoch % num_gpus].append(-1)
    i = 0
    for j in range(num_gpus):
        for k in range(len(partitions[j])):
            partitions[j][k] = i
            i += 1
    assert i == listlen
    for part in partitions:
        for each in part:
            assert each >= 0
    assert partitions[-1][-1] == listlen - 1
    return partitions


def deepcopy_cpu(x):
    return flor.common.copy.deepcopy(x)