from .cli import flags
from .versions import unpack


def main():
    if flags.args is not None and flags.args.flor_command is not None:
        if flags.args.flor_command == "unpack":
            unpack()
        elif flags.args.flor_command == "apply":
            from .hlast import apply

            apply(flags.args.dp_list, flags.args.train_file)


if __name__ == "__main__":
    main()
