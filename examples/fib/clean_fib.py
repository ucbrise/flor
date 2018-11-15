import flor
log = flor.log

@flor.track
def fib(idx):
    fib = {}
    fib[log.param(0)] = log.metric(0)
    fib[log.param(1)] = log.metric(1)
    fib[log.param(2)] = log.metric(2)

    for i in range(3, idx + 1):
        fib[log.param(i)] = log.metric(fib[i - 1] + fib[i - 2])


with flor.Context('fib'):
    fib(5)