from .clock import Clock
from .constants import *
from . import cli

import sys

if "ipykernel" not in sys.modules:
    cli.parse_args()

from .api import *
from .repl import query, dataframe, replay
from . import utils
from . import database

conn, cursor = database.conn_and_cursor()
database.create_tables(cursor)
conn.commit()
conn.close()
