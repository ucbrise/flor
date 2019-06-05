import flor
log = flor.log

@flor.track
def fib(idx):
    log.param(idx)
    if idx <= 2:
        return log.metric(idx)
    return log.metric(fib(idx - 1) + fib(idx - 2))

with flor.Context('fib'):
    fib(5)