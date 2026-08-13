from .clock import Clock
from .constants import *
from . import cli
from . import capture

from . import api
from .api import *
from .repl import query, dataframe, io, replay
from . import utils
from . import database

_interactive = False
try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        _interactive = True

        def _callback(*args):
            if output_buffer:
                commit()

        ip.events.register("post_run_cell", _callback)
    else:
        cli.parse_args()
except ImportError:
    cli.parse_args()


def _should_capture() -> bool:
    """Automatic io capture starts at import, not at first flor call.

    The whole point is the script that only does `import flordb` -- waiting for
    _deferred_init() would mean capturing nothing at all in that case.
    """
    if _interactive:
        # IPython swaps sys.stdout per cell and routes results through its own
        # displayhook, so a tee installed here would see the wrong stream and
        # duplicate cell output. Explicit flor.log is the interactive path.
        return False
    if SCRIPTNAME in ("-c", ""):
        # `python -c '...'` and the bare REPL. Capturing io is what registers a
        # run now (see api._register_run), so leaving these on would turn every
        # ad-hoc query into a recorded run with its own auto-commit.
        return False
    if cli.flags.args is not None and cli.flags.args.flor_command is not None:
        # `flor unpack` / `query` / `dataframe` / `replay` / `stat`: flor's own
        # CLI output, recording which would be the redundancy this feature is
        # named against.
        return False
    return True


if _should_capture():
    capture.install(api._emit_io)


conn, cursor = database.conn_and_cursor()
database.create_tables(cursor)
conn.commit()
conn.close()
