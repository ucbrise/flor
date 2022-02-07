from ast import arg
import json
import os
import pathlib
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import git
from git.repo import Repo

from flor.hlast import backprop


def parse_transform():
    parser = ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("lineno", type=int)
    return parser.parse_args(sys.argv[2:])


if sys.argv[1] == "transform":
    args = parse_transform()
    with open(".replay.json", "r") as f:
        name = json.load(f)["NAME"]
    dst = Path.home() / ".flor" / name / "repo.git"
    if dst.exists():
        shutil.rmtree(dst)
    transformed = Path.home() / ".flor" / name / "transformed"
    if not transformed.exists():
        transformed.mkdir()
    r = Repo()
    assert "flor.shadow" in str(r.active_branch)
    r.clone(dst)
    r = Repo(dst)
    commits = [
        c
        for c in r.iter_commits()
        if "flor.shadow" in c.message and ".json" == c.message[-len(".json") :]
    ]
    root = args.source.absolute()
    cwd = os.getcwd()
    os.chdir(dst)
    active = r.active_branch
    for version in commits:
        r.git.checkout(version)
        n = transformed / (str(version.hexsha) + "." + str(args.source))
        backprop(args.lineno, root, args.source, open(n, "w"))
    r.git.checkout(active)
    os.chdir(cwd)
