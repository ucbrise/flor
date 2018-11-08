import flor
log = flor.log

@flor.track_execution
def fib(idx):
    fib = {}
    fib[log.parameter(0)] = log.metric(0)
    fib[log.parameter(1)] = log.metric(1)
    fib[log.parameter(2)] = log.metric(2)

    for i in range(3, idx + 1):
        fib[log.parameter(i)] = log.metric(fib[i-1] + fib[i-2])


with flor.Context('fib'):
    fib(5)