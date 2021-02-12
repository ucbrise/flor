import file
import flags

"""
TODO: Add EOF Record to INDEX
"""

pid = flags.PID

current = start = 0
end = 20


def pre_training():
    return file.TREE.pre_training


def iterations_count():
    return file.TREE.iterations_count


def period():
    return file.TREE.period


def outermost_sk():
    return file.TREE.outermost_sk

