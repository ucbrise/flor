from .cli import flags
from .database import unpack

import json


def main():
    if flags.args is not None and flags.args.flor_command is not None:
        if flags.args.flor_command == "unpack":
            from . import versions

            start_branch = versions.current_branch()
            assert start_branch is not None
            for triplet in versions.get_latest_autocommit():
                print(triplet)
                _, next_commit, _ = triplet
                versions.checkout(next_commit)
                with open(".flor.json", "r") as f:
                    unpack(json.load(f))
            versions.checkout(start_branch.name)
        elif flags.args.flor_command == "apply":
            from .hlast import apply

            apply(flags.args.dp_list, flags.args.train_file)


if __name__ == "__main__":
    main()
