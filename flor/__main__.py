import argparse
from flor.commands.flython import exec_flython
from flor.commands.flan import exec_flan
from flor.commands.cp import exec_cp

def main(args=None):

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='subcommand')

    python_parser = subparsers.add_parser('python')
    python_parser.add_argument("path", help="The path to the stateful training pipeline to execute")
    python_parser.add_argument("name", help="The name of the experiment to run")
    python_parser.add_argument("-d", "--depth_limit", type=int, help="Depth limit the logging")

    etl_parser = subparsers.add_parser('etl')
    etl_parser.add_argument("name", help="The name of the experiment that has been run")
    etl_parser.add_argument("annotated_file", nargs='+', help="The paths to the annotated files")
    etl_parser.add_argument("-d", "--dump_db", choices={"postgres", "sqlite"}, help="Load data to db")
    etl_parser.add_argument("-f", "--dump_file", help="Load data to file at specified path")

    cp_parser = subparsers.add_parser('cp')
    cp_parser.add_argument("src", help="The path to the source file that will be annotated")
    cp_parser.add_argument("dst", help="Name of the file that will be copied")

    if args is None:
        args = parser.parse_args()

    try:
        if args.subcommand == 'python':
            exec_flython(args)
        elif args.subcommand == 'etl':
            exec_flan(args)
        elif args.subcommand == 'cp':
            exec_cp(args)
        else:
            raise ValueError("Invalid option: {}".format(args.subcommand))
    except:
        pass

if __name__ == '__main__':
    main()