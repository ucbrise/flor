from ast import arg
import json
import os
import pathlib
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import git
from git.repo import Repo

from flor.hlast import backprop


def parse_transform():
    parser = ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("lineno", type=int, nargs="?", default=None)
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
        if "flor.shadow" in str(c.message) and ".json" == c.message[-len(".json") :]
    ]
    root = args.source.absolute()
    cwd = os.getcwd()
    os.chdir(dst)
    active = r.active_branch
    for version in commits:
        r.git.checkout(version)
        n = transformed / (str(version.hexsha) + "." + str(args.source))
        try:
            backprop(None, root, args.source, open(n, "w"))  # type: ignore
            print(f'transformed {(str(version.hexsha) + "::" + str(args.source))}')
        except FileNotFoundError:
            # print(f"version {version.hexsha[0:6]}... does not contain {args.source}")
            os.remove(n)
            continue
        except:
            os.remove(n)
            continue

    r.git.checkout(active)
    os.chdir(cwd)
elif sys.argv[1] == "unpack":
    with open(".replay.json", "r") as f:
        name = json.load(f)["NAME"]
    dst = Path.home() / ".flor" / name / "repo.git"
    if dst.exists():
        shutil.rmtree(dst)
    replay_jsons = Path.home() / ".flor" / name / "replay_jsons"
    if not replay_jsons.exists():
        replay_jsons.mkdir()
    r = Repo()
    assert "flor.shadow" in str(r.active_branch)
    r.clone(dst)
    r = Repo(dst)
    commits = [
        c
        for c in r.iter_commits()
        if "flor.shadow" in str(c.message) and ".json" == c.message[-len(".json") :]
    ]
    cwd = os.getcwd()
    os.chdir(dst)
    active = r.active_branch  # check behavior
    for version in commits:
        r.git.checkout(version)
        hexsha, message = version.hexsha, version.message
        _, tstamp_json = message.split("::")
        try:
            shutil.copy2(".replay.json", os.path.join(replay_jsons, tstamp_json))  # type: ignore
            print(f'copied {(str(version.hexsha) + "::" + str(tstamp_json))}')
        except FileNotFoundError:
            # print(f"version {version.hexsha[0:6]}... does not contain {args.source}")
            continue
        except:
            continue

    r.git.checkout(active)
    os.chdir(cwd)
elif sys.argv[1] == "stage":
    args = parse_transform()
    with open(".replay.json", "r") as f:
        name = json.load(f)["NAME"]
    dst = Path.home() / ".flor" / name / "staged.py"
    assert args.source.exists()
    shutil.copy2(args.source, dst)
    print(f"{args.source} staged!")
elif sys.argv[1] == "propagate":
    args = parse_transform()
    with open(".replay.json", "r") as f:
        name = json.load(f)["NAME"]
    src = Path.home() / ".flor" / name / "staged.py"
    assert src.exists(), "Did you first stage the file you want to propagate from?"
    backprop(None, src, target=args.source)  # type: ignore
    print(f"transformed {str(args.source)}")
    print(f"{args.source} modified to include logging statements")
