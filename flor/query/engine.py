from typing import List
from pathlib import Path


def replay(logged_vars: List[str], where_clause: str, path: str):
    """
    logged_vars : ['device', 'optimizer', 'learning_rate', ...]
    path: `train_rnn.py` or such denoting main python script
    where_clause: stated in Pandas/SQL, passed to full_pivot
    """
    assert Path(path).suffix == '.py'
