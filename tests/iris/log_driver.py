from flor.log_scanner.scanners.actual_param import ActualParam
from flor.log_scanner.scanner import Scanner
import json

scanner = Scanner('log.json')

scanner.register_state_machine(ActualParam("/Users/rogarcia/sandbox/iris/iris_raw.py",
                           None,
                           None,
                           2,
                           3,
                           'train_test_split',
                           {'pos': 0}))

scanner.register_state_machine(ActualParam("/Users/rogarcia/sandbox/iris/iris_raw.py",
                           None,
                           None,
                           2,
                           3,
                           'train_test_split',
                           {'kw': 'test_size'}))

scanner.register_state_machine(ActualParam("/Users/rogarcia/sandbox/iris/iris_raw.py",
                              None,
                              None,
                              2,
                              3,
                              'train_test_split',
                              {'kw': 'random_state'}))

scanner.register_state_machine(ActualParam("/Users/rogarcia/sandbox/iris/iris_raw.py",
                              None,
                              None,
                              3,
                              4,
                              'SVC',
                              {'kw': 'gamma'}))

scanner.register_state_machine(ActualParam("/Users/rogarcia/sandbox/iris/iris_raw.py",
                              None,
                              None,
                              3,
                              4,
                              'SVC',
                              {'kw': 'C'}))

scanner.scan_log()

print("collected: {}".format(scanner.collected))