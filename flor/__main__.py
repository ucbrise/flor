import argparse
import subprocess

from flor.commands.flython import exec_flython
from flor.commands.flan import exec_flan
from flor.commands.cp import exec_cp
import sys
from flor.complete_capture.walker import Walker
from flor import utils
from flor.constants import *
import pip


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

    if args.subcommand == 'python':
        exec_flython(args)
    elif args.subcommand == 'etl':
        exec_flan(args)
    elif args.subcommand == 'cp':
        exec_cp(args)
    else:
        raise ValueError("Invalid option: {}".format(args.subcommand))


def uninstall(multiuser=False):
    """
    Completely uninstall Flor: Removes the pyflor package, the conda
    environment named conda_flor_env, and everything except logs in
    FLOR_DIR.
    """
    utils.check_flor_install()

    from flor.constants import FLOR_DIR

    if multiuser and os.path.isdir(FLOR_SHARE):
        FLOR_DIR = FLOR_SHARE

    conda_map_path = os.path.join(FLOR_DIR, '.conda_map')

    with open(conda_map_path, 'r') as f:
        src_root, dst_root = f.read().strip().split(',')

        conda_flor_env = dst_root.split('/')[-1]

        subprocess.call(['conda', 'remove', '--name', conda_flor_env, '--all'])

        subprocess.call((['pip', 'uninstall', 'pyflor']))

        os.remove(conda_map_path)


def install(base_conda='base', conda_flor_env='flor', python_version='3.7', multiuser=False):
    def make_conda_map(d, base_env_path, env_path):
        utils.cond_mkdir(d)
        with open(os.path.join(d, '.conda_map'), 'w') as f:
            # Storing the map src conda -> dst conda
            # This will be used for flor cp, so the user sees the pre-transformed code.
            f.write(base_env_path + ',' + env_path + '\n')

    if multiuser and os.path.isdir(FLOR_SHARE):
        # initialize flor in current directory but do nothing else
        with open(os.path.join(FLOR_SHARE, '.conda_map'), 'r') as f:
            src_root, dst_root = f.read().strip().split(',')
        make_conda_map(FLOR_DIR, src_root, dst_root)
        return

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

    # conda clone environment
    subprocess.call(['conda', 'create', '--name', conda_flor_env, '--clone', base_conda])

    raw_envs = subprocess.check_output(['conda', 'info', '--envs']).decode('utf-8')
    raw_envs = raw_envs.split('\n')
    raw_envs = raw_envs[2:]

    env_path = None
    base_env_path = None
    for raw_env in raw_envs:
        raw_env = raw_env.replace('*', '')
        name_path = raw_env.split()
        if len(name_path) != 2:
            continue
        name, path = name_path
        if name == conda_flor_env:
            env_path = path
        elif name == base_conda:
            base_env_path = path
    assert env_path is not None
    assert base_env_path is not None

    multiuser and make_conda_map(FLOR_SHARE, base_env_path, env_path)
    make_conda_map(FLOR_DIR, base_env_path, env_path)

    env_path = os.path.join(env_path, 'lib', 'python' + python_version, 'site-packages')

    # Perform lib transformation
    walker = Walker(env_path)
    walker.compile_tree()

    print("Install succeeded.")

    # Add Flor script to user shell config file
    shells_list = ['.zshrc', '.bashrc', '.cshrc', '.kshrc', '.config/fish/config.fish']
    for s in shells_list:
        shell_config = os.path.join(os.path.expanduser('~'), s)
        if os.path.exists(shell_config) and FLOR_FUNC not in open(shell_config, 'r').read():
            with open(shell_config, 'a') as f:
                f.write(FLOR_FUNC)

    print("Please append the following lines to your shell configuration file:\n"
          "" + FLOR_FUNC)


if __name__ == '__main__':
    main()
