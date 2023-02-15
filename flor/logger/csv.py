from flor.shelf import home_shelf
from flor.state import State
from typing import Dict, List

import pandas as pd

class CSV_Writer:
    def __init__(self, name, columns) -> None:
        self.name = name
        self.cols = columns
        self.records: List[Dict[str, str]] = []

    def put(self, values):
        self.records.append({col: v for col, v in zip(self.cols, values)})

    def extend(self, batch):
        self.records.extend([{col:v} for values in batch for col, v in zip(self.cols, values) ])

    def flush(self):
        path = home_shelf.get_csv_ref(self.name, State.timestamp)
        assert path is not None
        pd.DataFrame(self.records).to_csv(path, index=False)
    