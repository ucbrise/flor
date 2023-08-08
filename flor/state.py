from argparse import Namespace
from typing import Optional

args: Optional[Namespace] = None
hyperparameters = {}


def replay_mode():
    assert args is not None
    if args.replay_flor is None:
        return False
    else:
        return True
