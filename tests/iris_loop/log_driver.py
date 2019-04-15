from flor.log_scanner.scanners.actual_param import ActualParam
import json

ap_training_data = ActualParam("/Users/rogarcia/sandbox/iris_loop/iris_raw.py",
                               None,
                               None,
                               2,
                               3,
                               'train_test_split',
                               {'pos': 0})

ap_test_size = ActualParam("/Users/rogarcia/sandbox/iris_loop/iris_raw.py",
                           None,
                           None,
                           2,
                           3,
                           'train_test_split',
                           {'kw': 'test_size'})

ap_random_state = ActualParam("/Users/rogarcia/sandbox/iris_loop/iris_raw.py",
                              None,
                              None,
                              2,
                              3,
                              'train_test_split',
                              {'kw': 'random_state'})

ap_gamma = ActualParam("/Users/rogarcia/sandbox/iris_loop/iris_raw.py",
                              None,
                              None,
                              4,
                              6,
                              'SVC',
                              {'kw': 'gamma'})

ap_C = ActualParam("/Users/rogarcia/sandbox/iris_loop/iris_raw.py",
                              None,
                              None,
                              4,
                              6,
                              'SVC',
                              {'kw': 'C'})

aps = [ap_training_data, ap_test_size, ap_random_state, ap_gamma, ap_C]
with open('log.json', 'r') as f:
    for i, line in enumerate(f):
        d = json.loads(line.strip())
        for ap in aps:
            ap.transition(d)
        # VERIFICATION CODE
        trailing_ctx, contexts =  aps[0].trailing_ctx, aps[0].contexts
        for ap in aps:
            try:
                assert ap.trailing_ctx == trailing_ctx
            except:
                print('error at iteration {} difference: {} ... {}'.format(i, ap.trailing_ctx, ap.contexts))

collected = []
for ap in aps:
    collected += ap.collected
print("collected: {}".format(collected))