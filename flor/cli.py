import argparse
from . import state


def parse_args():
    parser = argparse.ArgumentParser(description="Replay flor script")
    parser.add_argument(
        "--replay_flor",
        nargs="*",
        type=lambda s: int(s) if s != "/" else s,
        help='Use "--replay_flor PID / NGPUS" format where PID and NGPUS are integers, and "/" is a literal',
    )

    # Collect additional key-value pair arguments
    parser.add_argument(
        "--kwargs",
        nargs="*",
        type=str,
        help="Additional key-value pair arguments for hyper-parameters",
    )

    args = parser.parse_args()
    state.args = args

    # Process the key-value pair arguments
    if args.kwargs:
        for kwarg in args.kwargs:
            key, value = kwarg.split("=")
            state.hyperparameters[key] = value

    return args, hyperparameters


if __name__ == "__main__":
    args, hyperparameters = parse_args()

    if args.replay_flor is None:
        print("Default mode")
    elif not args.replay_flor:
        print("Flor replay mode")
    elif len(args.replay_flor) == 3 and args.replay_flor[1] == "/":
        print(
            f"Flor replay with PID: {args.replay_flor[0]}, NGPUS: {args.replay_flor[2]}"
        )
    else:
        print("Invalid input for --replay_flor")

    print("Hyperparameters:", hyperparameters)
