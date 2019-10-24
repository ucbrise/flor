import os

def pure_func(f):
    def fork_protect(*args, **kwargs):
        pid = os.fork()
        if not pid:
            try:
                f(*args, **kwargs)
            finally:
                os._exit(0)