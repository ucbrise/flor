#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any, Set
import json

from flor import global_state

stack_frame: List[Tuple[str, str]] = []
consume_from = {}
dict_of_loopids = {}

# Maps a value that was returned by a function to the names of the functions that return that value
dict_of_returns: Dict[Any, Set[str]] = {}

file = None

log_record_buffer = []
log_record_flag = False


class FlorEnter:
    pass

class FlorExit:
    pass

def internal_log(v, d):
    """
    Creates a new LOG_RECORD
    :param v:
    :param d:
    :return:
    """
    d['runtime_value'] = v
    d['__stack_frame__'] = tuple(stack_frame)
    if log_record_flag:
        log_record_buffer.append(d)
    file.write(json.dumps(d, indent=4) + ',\n')
    return v

def log_enter(locl=None, vararg=None, kwarg=None, func_name=None, iteration_id=None):
    """
    Signals the entry of a BLOCK_NODE
    :param locl: a call to locals(). A dictionary mapping name to value of input args to a function
    :param vararg:
    :param kwarg:
    :param func_name:
    :param iteration_id: If entering loop body block, iteration_id is an integer
    :return:
    """
    if func_name:
        # block_type: function_body
        function_identifier = ('function_body', func_name)
        stack_frame.append(function_identifier)

        varargs = []
        kwargs = []

        consumes = False

        if locl is not None:
            if vararg is not None:
                varargs = list(locl[vararg])
                del locl[vararg]
            if kwarg is not None:
                kwargs = list(locl[kwarg].values())
                del locl[kwarg]

            for arg in list(locl.values()) + varargs + kwargs:
                for returned_value in dict_of_returns:
                    func_names = dict_of_returns[returned_value]
                    consumes = returned_value is arg
                    if type(returned_value) == list or type(returned_value) == tuple:
                        consumes = consumes or any([x is arg for x in returned_value])
                    if consumes:
                        if func_name not in consume_from or log_record_flag:
                            consume_from[func_name] = set(["{}".format(each) for each in func_names])
                            # write fact to log first time it appears
                            d = {'to': func_name, 'from': tuple(consume_from[func_name])}
                            if log_record_flag:
                                log_record_buffer.append(d)
                            file.write(json.dumps(d, indent=4) + ',\n')

                        else:
                            assert consume_from[func_name] == set(["{}".format(each) for each in func_names])
                        break
                if consume_from:
                    break
    else:
        # block_type: loop_body
        assert iteration_id is not None
        stack_identifier = tuple(stack_frame)
        iteration_num = 0
        if stack_identifier not in dict_of_loopids:
            dict_of_loopids[stack_identifier] = {iteration_id: 0}
        elif iteration_id not in dict_of_loopids[stack_identifier]:
            dict_of_loopids[stack_identifier][iteration_id] = 0
        else:
            dict_of_loopids[stack_identifier][iteration_id] += 1
            iteration_num = dict_of_loopids[stack_identifier][iteration_id]

        stack_frame.append(('loop_body', str(iteration_num)))


def log_exit(v=None, is_function=False):
    """
    Signals the exit of a BLOCK_NODE
    :param v:
    :return:
    """

    if v:
        # Return Context
        if v in dict_of_returns:
            dict_of_returns[v] |= {stack_frame[-1][1],}
        else:
            dict_of_returns[v] = {stack_frame[-1][1],}

    if is_function and global_state.interactive:
        global file
        file.close()
        file = open(global_state.log_name, 'a')

    stack_frame.pop()

    return v