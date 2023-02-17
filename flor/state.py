from typing import Optional
from pathlib import Path
from git.repo import Repo


class State:
    loop_nesting_level = 0
    epoch: Optional[int] = None
    step: Optional[int] = None
    timestamp: Optional[str] = None
    repo: Optional[Repo] = None
    common_dir: Optional[Path] = None
    active_branch: Optional[str] = None