import os

def pure_func(f):
    def fork_protect(*args, **kwargs):
        pid = os.fork()
        if not pid:
            try:
                f(*args, **kwargs)
            except:
                import sys
                import traceback
                e = sys.exc_info()[0]
                traceback.print_exc()
                print(e)
            finally:
                os._exit(0)
    return fork_protect