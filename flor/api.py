import utils
from contextlib import contextmanager
from . import state


layers = []
checkpoints = []


def log(name, value):
    serializable_value = value if utils.is_jsonable(value) else str(value)
    print(name, serializable_value)
    # if State.loop_nesting_level:
    #     log_records.put(name, serializable_value)
    # else:
    #     exp_json.put(name, serializable_value, ow=False)
    return value


def arg(name, default=None):
    if state.replay_mode():
        # GIT
        pass
    elif name in state.hyperparameters:
        # CLI
        v = state.hyperparameters[name]
        if default is not None:
            return utils.duck_cast(v, default)
        return v
    elif default is not None:
        return default
    else:
        raise


@contextmanager
def checkpointing(*args):
    # set up the context
    layers.extend(list(args))

    yield  # The code within the 'with' block will be executed here.

    # tear down the context if needed
    layers[:] = []
    checkpoints[:] = []


def layer(name, iterator):
    pass
