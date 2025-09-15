from .clock import Clock
from .constants import *
from . import cli

from .api import *
from .repl import query, dataframe, replay
from . import utils
from . import database

try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:

        def _callback(*args):
            if output_buffer:
                commit()

        ip.events.register("post_run_cell", _callback)
    else:
        cli.parse_args()
except ImportError:
    cli.parse_args()


conn, cursor = database.conn_and_cursor()
database.create_tables(cursor)
conn.commit()
conn.close()
