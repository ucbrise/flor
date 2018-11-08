import flor
log = flor.log

@flor.track_execution
def fib(idx):
    log.parameter(idx)
    if idx <= 2:
        return idx
    return log.metric(fib(idx - 1) + fib(idx - 2))

with flor.Context('fib'):
    fib(5)