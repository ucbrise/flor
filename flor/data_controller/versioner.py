#!/usr/bin/env python3

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from flor.experiment_graph import ExperimentGraph
    from flor.stateful import State

import git
import tempfile
import os
import cloudpickle

from shutil import rmtree
from shutil import move
from shutil import copy2 as copy

#

class Versioner:
    """
    Responsible for putting Literal and Code artifacts in ~/flor.d
    """

    def __init__(self, version, eg: 'ExperimentGraph', xp_state: 'State'):
        self.eg = eg
        self.xp_state = xp_state
        self.original = os.getcwd()
        self.versioning_dir = os.path.join(self.xp_state.versioningDirectory, self.xp_state.EXPERIMENT_NAME)
        self.version = version

    @staticmethod
    def __is_git_repo__(path):
        # https://stackoverflow.com/a/39956572/9420936
        try:
            _ = git.Repo(path).git_dir
            return True
        except:
            return False

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

    def __move_starts__(self):
        for start in self.eg.starts:
            #TODO: Generalize. What if we have non-python scripts or files
            #TODO: Will need to generalize by typing flor Artifacts, propagate to ArtifactLight
            if type(start).__name__[0:len('Artifact')] == 'Artifact':
                start = start.loc
                if start.split('.')[-1] in ('py', 'ipynb'):
                    copy(start, os.path.join(self.versioning_dir, start))
        copy('experiment_graph.pkl', os.path.join(self.versioning_dir, 'experiment_graph.pkl'))
        if os.path.exists('output.gv'):
            copy('output.gv', os.path.join(self.versioning_dir, 'output.gv'))
        if os.path.exists('output.gv.pdf'):
            copy('output.gv.pdf', os.path.join(self.versioning_dir, 'output.gv.pdf'))
        copy(self.xp_state.florFile, os.path.join(self.versioning_dir, self.xp_state.florFile))

    def __serialize_literals__(self):
        literals = self.eg.get_literals_reachable_from_starts()
        assert (all([type(lit).__name__ == 'LiteralLight' for lit in literals]))

        cloud_serializable_literals = filter(lambda lit: not Versioner.__is_string_serializable__(lit.v), literals)

        for lit in cloud_serializable_literals:
            lit_name = "{}_{}.pkl".format(lit.name, id(lit))
            with open(os.path.join(self.versioning_dir, lit_name), 'wb') as f:
                cloudpickle.dump(lit.v, f)


    def __git_commit__(self, mode, msg):
        os.chdir(self.versioning_dir)
        if mode == 'initial':
            repo = git.Repo.init(os.getcwd())
            repo.git.add(A=True)
            repo.index.commit(msg)
        else:
            repo = git.Repo(os.getcwd())
            repo.git.add(A=True)
            repo.index.commit(msg)
        os.chdir(self.original)


    def save_commit_event(self):
        """
        On building flor-plan, versions all the code artifacts
        """
        if os.path.exists(self.versioning_dir):
            with tempfile.TemporaryDirectory() as tempdir:
                move(os.path.join(self.versioning_dir, '.git'), tempdir)
                #TODO: optimize, this remove results in redundant copy
                rmtree(self.versioning_dir)
                os.mkdir(self.versioning_dir)
                self.__move_starts__()
                move(os.path.join(tempdir, '.git'), os.path.join(self.versioning_dir, '.git'))
            self.__git_commit__('incremental', "msg:commit:{}".format(self.version))
        else:
            if not os.path.exists(self.xp_state.versioningDirectory):
                os.mkdir(self.xp_state.versioningDirectory)
            os.mkdir(self.versioning_dir)
            self.__move_starts__()
            self.__git_commit__('initial', "msg:commit:{}".format(self.version))

    def save_pull_event(self):
        """
        Artifacts: Move all code artifacts again, don't move anything from the organizer
          Disjoint set. Superset of save_commit_event
        Literals: Serializable in str() put in GROUND; Else go in cloudPickle Git, and GROUND
          Links to it
        """
        #TODO: Return commit hash after most recent commit
        assert self.__is_git_repo__(self.versioning_dir), "Invariant Violation: Pull event without prior commit event"
        # self.versioning_dir exists
        with tempfile.TemporaryDirectory() as tempdir:
            move(os.path.join(self.versioning_dir, '.git'), tempdir)
            # TODO: optimize, this remove results in redundant copy
            rmtree(self.versioning_dir)
            os.mkdir(self.versioning_dir)
            self.__move_starts__()
            self.__serialize_literals__()
            move(os.path.join(tempdir, '.git'), os.path.join(self.versioning_dir, '.git'))
        self.__git_commit__('incremental', "msg:pull:{}".format(self.version))
