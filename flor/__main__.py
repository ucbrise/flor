from .constants import *
from .cli import flags
from .database import unpack, create_tables

import json
import os
from pathlib import Path


def main():
    if flags.args is not None and flags.args.flor_command is not None:
        if flags.args.flor_command == "unpack":
            from . import versions
            import sqlite3

            connection = sqlite3.connect(
                os.path.join(HOMEDIR, Path(PROJID).with_suffix(".db"))
            )
            cursor = connection.cursor()
            create_tables(cursor)

            start_branch = versions.current_branch()
            assert start_branch is not None
            try:
                for triplet in versions.get_latest_autocommit():
                    _, next_commit, _ = triplet
                    versions.checkout(next_commit)
                    with open(".flor.json", "r") as f:
                        unpack(json.load(f), cursor)
                connection.commit()
                connection.close()
            finally:
                versions.checkout(start_branch.name)
        elif flags.args.flor_command == "apply":
            from .hlast import apply

            apply(flags.args.dp_list, flags.args.train_file)


if __name__ == "__main__":
    main()
