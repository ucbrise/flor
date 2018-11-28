#!/usr/bin/env python3

import git
import os


class Versioner:

    def __init__(self):
        self.current_dir = os.getcwd()
        self.versioning_dir = Versioner.get_ancestor_repo_path(self.current_dir)


    @staticmethod
    def __is_git_repo__(path):
        # https://stackoverflow.com/a/39956572/9420936
        try:
            _ = git.Repo(path).git_dir
            return True
        except:
            return False

    @staticmethod
    def get_ancestor_repo_path(path):
        if not path or path == os.path.sep:
            return None
        path = os.path.abspath(path)
        if Versioner.__is_git_repo__(path):
            return path
        head, tail = os.path.split(path)
        return Versioner.get_ancestor_repo_path(head)

    @staticmethod
    def __is_string_serializable__(x):
        if type(x) == str:
            return True
        try:
            str_x = str(x)
            x_prime = eval(str_x)
            assert x_prime == x
            return True
        except:
            return False

    def save_commit_event(self, msg, log_file):
        if not self.versioning_dir:
            # TODO: ADD GIT_IGNORE FOR THE USER
            repo = git.Repo.init(self.current_dir)
            repo.git.add(A=True)
            repo.git.add([log_file], force=True)
            repo.index.commit(msg)
        else:
            repo = git.Repo(self.versioning_dir)
            repo.git.add(A=True)
            repo.git.add([log_file], force=True)
            repo.index.commit(msg)

