from typing import Optional


class State:
    loop_nesting_level = 0
    epoch: Optional[int] = None
    step: Optional[int] = None
    timestamp: Optional[str] = None
