import json

from flor.controller.parser.injected import structured_log

class Context():
    def __init__(self, xp_name):
        self.xp_name = xp_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # str_serial = ""
        #
        # for log in log_sequence:
        #     if log is FlorEnter:
        #         str_serial += "["
        #     elif log is FlorExit:
        #         str_serial += "], "
        #     else:
        #         str_serial += str(log) + ", "
        #
        # eval_serial, *_ = eval(str_serial)
        #
        # with open('{}_log.json'.format(self.xp_name), 'w') as f:
        #     json.dump(eval_serial, f)

        with open('{}_log.json'.format(self.xp_name), 'w') as f:
            json.dump(structured_log.log_tree, f)