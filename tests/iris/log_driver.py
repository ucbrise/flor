from flor.log_scanner.scanners.actual_param import ActualParam
import json

ap_rando = ActualParam("/Users/rogarcia/sandbox/iris/iris_raw.py",
                           None,
                           None,
                           2,
                           3,
                           'train_test_split',
                           {'pos': 0})

ap_test_size = ActualParam("/Users/rogarcia/sandbox/iris/iris_raw.py",
                           None,
                           None,
                           2,
                           3,
                           'train_test_split',
                           {'kw': 'test_size'})

ap_random_state = ActualParam("/Users/rogarcia/sandbox/iris/iris_raw.py",
                              None,
                              None,
                              2,
                              3,
                              'train_test_split',
                              {'kw': 'random_state'})

ap_gamma = ActualParam("/Users/rogarcia/sandbox/iris/iris_raw.py",
                              None,
                              None,
                              3,
                              4,
                              'SVC',
                              {'kw': 'gamma'})

ap_C = ActualParam("/Users/rogarcia/sandbox/iris/iris_raw.py",
                              None,
                              None,
                              3,
                              4,
                              'SVC',
                              {'kw': 'C'})

# aps = [ap_gamma,]
aps = [ap_rando, ap_test_size, ap_random_state, ap_gamma, ap_C]
with open('log.json', 'r') as f:
    for line in f:
        d = json.loads(line.strip())
        for ap in aps:
            ap.transition(d)

collected = []
for ap in aps:
    collected += ap.collected
print("collected: {}".format(collected))