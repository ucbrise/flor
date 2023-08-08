from argparse import Namespace
from typing import Dict, Optional

args: Optional[Namespace] = None
hyperparameters: Dict[str, str] = {}


def replay_mode():
    assert args is not None
    return args.replay_flor is not None
