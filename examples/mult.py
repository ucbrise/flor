import jarvis

@jarvis.func
def multiply(x, y):
    z = int(x) * int(y)
    print(z)
    return z