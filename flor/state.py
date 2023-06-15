from typing import Dict, List, Optional, Union
from pathlib import Path
from git.repo import Repo
import sqlite3


class State:
    loop_nesting_level = 0
    epoch: Optional[int] = None
    step: Optional[int] = None
    timestamp: Optional[str] = None
    repo: Optional[Repo] = None
    common_dir: Optional[Path] = None
    active_branch: Optional[str] = None
    db_conn: Optional[sqlite3.Connection] = None

    import_time: Optional[float] = None
    seconds: Dict[str, Optional[Union[float, List[float]]]] = {
        "PREP": None,
    }  # May add "EPOCHS" at runtime
    target_block = None
    
    hls_hits = set([]) # hindsight logging stmt hits
    grouped_names: Dict[str, int] = {}

    @staticmethod
    def get_vid():
        if State.repo is not None:
            for v in State.repo.iter_commits():
                if str(v.message).count("RECORD::") == 1:
                    return v.hexsha
