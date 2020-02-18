import math
import os
import shutil
import flor.common.copy
import copy


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


def deepcopy_cpu(x):
    copy.deepcopy = flor.common.copy.deepcopy
    return copy.deepcopy(x)

def has_method(x, name):
    return hasattr(x, name) and callable(getattr(x, name))

def copy_for_store(x):
    if has_method(x, 'state_dict'):
        return deepcopy_cpu(x.state_dict())
    elif hasattr(x, 'cpu') and callable(getattr(x, 'cpu')):
        return x.cpu()
    try:
        return deepcopy_cpu(x)
    except:
        attr_val_dict = {}
        for attr_name in x.__dict__.keys():
            attr_obj = getattr(x, attr_name)
            if has_method(attr_obj, 'state_dict'):
                attr_val_dict[attr_name] = deepcopy_cpu(attr_obj.state_dict())
            elif has_method(attr_obj, 'cpu'):
                attr_val_dict[attr_name] = x.cpu()
            else:
                try:
                    attr_val_dict[attr_name] = deepcopy_cpu(attr_obj)
                except Exception:
                    pass
            attr_val_dict['_flor_stored_by_dict'] = True
        return attr_val_dict

def load_by_dict(x, attr_val_dict):
    attr_val_dict.pop('_flor_stored_by_dict')
    for attr_name in attr_val_dict:
        attr_obj = getattr(x, attr_name)
        val = attr_val_dict[attr_name]
        if has_method(attr_obj, 'state_dict'):
            attr_obj.load_state_dict(val)
        else:
            setattr(x, attr_name, val)
