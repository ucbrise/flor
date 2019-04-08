import argparse
import os

from flor.versioner.versioner import Versioner

parser = argparse.ArgumentParser()
parser.add_argument("path", help="The path to the model training pipeline to execute")

args = parser.parse_args()

if __name__ == '__main__':
    full_path = os.path.abspath(args.path)
    versioner = Versioner(full_path)
    versioner.save_commit_event("flor commit")