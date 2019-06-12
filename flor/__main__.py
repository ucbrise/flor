import argparse
from flor.commands.flython import exec_flython
from flor.commands.flan import exec_flan
from flor.commands.cp import exec_cp
import sys
import shutil
from flor.complete_capture.walker import Walker
from flor import utils
from flor.constants import *

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

def install(base_conda='base', conda_flor_env='flor', python_version='3.7'):
    import pip

    def _install(package):
        if hasattr(pip, 'main'):
            pip.main(['install', package])
        else:
            pip._internal.main(['install', package])

    try:
        import conda.cli.python_api
    except ImportError:
        _install('conda')
        import conda.cli.python_api

    print("The Flor installer needs to copy and transform an anaconda environment.\n"
          "This installer will take some time to complete.")

    while True:
        res = input("Continue with installation [Y/n]? ").strip().lower()
        if res == 'n':
            print("Exiting...")
            sys.exit(0)
        if res == 'y' or not res:
            break
        print("Invalid response: {}".format(res))

    base_conda = input("Enter the source anaconda environment [{}]: ".format(base_conda)).strip() or base_conda
    python_version = input("Enter the Python version of the source anaconda environment [{}]: ".format(python_version)) or python_version
    conda_flor_env = input("Enter the name of the new anaconda environment [{}]: ".format(conda_flor_env)).strip() or conda_flor_env

    FLOR_FUNC = """
    flor() {{
            conda activate {};
            pyflor $@;
            cd $(pwd);
            conda deactivate;
    }}
    """.format(conda_flor_env)

    conda.cli.python_api.run_command('create', '--name', conda_flor_env, '--clone', base_conda)

    raw_envs, _, _ = conda.cli.python_api.run_command('info', '--envs')
    raw_envs = raw_envs.split('\n')
    raw_envs = raw_envs[2:]

    env_path = None
    base_env_path = None
    for raw_env in raw_envs:
        raw_env = raw_env.replace('*', '')
        name_path = raw_env.split()
        if len(name_path) < 2:
            continue
        name, path = name_path
        if name == conda_flor_env:
            env_path = path
        elif name == base_conda:
            base_env_path = path
    assert env_path is not None
    assert base_env_path is not None

    utils.cond_mkdir(FLOR_DIR)
    with open(os.path.join(FLOR_DIR, '.conda_map'), 'w') as f:
        # Storing the map src conda -> dst conda
        # This will be used for flor cp, so the user sees the pre-transformed code.
        f.write(base_env_path + ',' + env_path + '\n')

    env_path = os.path.join(env_path, 'lib', 'python' + python_version, 'site-packages')

    walker = Walker(env_path)
    walker.compile_tree()

    shutil.rmtree(env_path)
    shutil.move(walker.targetpath, env_path)

    print("Install succeeded.")

    print("Please append the following line to your shell configuration file:\n"
          "" + FLOR_FUNC)


if __name__ == '__main__':
    main()