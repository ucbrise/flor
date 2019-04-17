import argparse
from flython import exec_flython
from flan import exec_flan

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest='subcommand')

python_parser = subparsers.add_parser('python')
python_parser.add_argument("path", help="The path to the model training pipeline to execute")
python_parser.add_argument("name", help="The name of the experiment to run")
python_parser.add_argument("-d", "--depth_limit", type=int, help="Depth limit the logging")

etl_parser = subparsers.add_parser('etl')
etl_parser.add_argument("name", help="The name of the experiment that has been run")
etl_parser.add_argument("annotated_file", nargs='+', help="The paths to the annotated files")
etl_parser.add_argument("-d", "--dump_db", choices={"postgres", "sqlite"}, help="Load data to db")
etl_parser.add_argument("-f", "--dump_file", help="Load data to file at specified path")

args = parser.parse_args()

if __name__ == '__main__':
    if args.subcommand == 'python':
        exec_flython(args)
    else:
        exec_flan(args)